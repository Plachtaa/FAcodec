import shutil
import warnings
import argparse
import torch
import os
import os.path as osp
import yaml

warnings.simplefilter('ignore')

# load packages
import random

from meldataset import build_dataloader
from modules.commons import *
from losses import *
from optimizers import build_optimizer
import time

from accelerate import Accelerator
from accelerate.utils import LoggerType
from accelerate import DistributedDataParallelKwargs

from torch.utils.tensorboard import SummaryWriter
import torchaudio
# from torchmetrics.classification import MulticlassAccuracy

import logging
from accelerate.logging import get_logger

from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from dac.nn.loss import MultiScaleSTFTLoss, MelSpectrogramLoss, GANLoss, L1Loss
from audiotools import AudioSignal

import nemo.collections.asr as nemo_asr
import glob


logger = get_logger(__name__, log_level="INFO")
# torch.autograd.set_detect_anomaly(True)

def main(args):
    config_path = args.config_path
    config = yaml.safe_load(open(config_path))

    log_dir = config['log_dir']
    if not osp.exists(log_dir): os.makedirs(log_dir, exist_ok=True)
    shutil.copy(config_path, osp.join(log_dir, osp.basename(config_path)))
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True, broadcast_buffers=True)
    accelerator = Accelerator(project_dir=log_dir, split_batches=True, kwargs_handlers=[ddp_kwargs])
    if accelerator.is_main_process:
        writer = SummaryWriter(log_dir + "/tensorboard")

    # write logs
    file_handler = logging.FileHandler(osp.join(log_dir, 'train.log'))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(levelname)s:%(asctime)s: %(message)s'))
    logger.logger.addHandler(file_handler)

    batch_size = config.get('batch_size', 10)
    batch_length = config.get('batch_length', 120)
    device = accelerator.device if accelerator.num_processes > 1 else torch.device('cpu')

    epochs = config.get('epochs', 200)
    log_interval = config.get('log_interval', 10)
    saving_epoch = config.get('save_freq', 2)
    save_interval = config.get('save_interval', 1000)

    data_params = config.get('data_params', None)
    sr = config['preprocess_params'].get('sr', 24000)
    train_path = data_params['train_data']
    val_path = data_params['val_data']
    root_path = data_params['root_path']
    max_frame_len = config.get('max_len', 80)
    discriminator_iter_start = config['loss_params'].get('discriminator_iter_start', 0)
    loss_params = config.get('loss_params', {})
    hop_length = config['preprocess_params']['spect_params'].get('hop_length', 300)
    win_length = config['preprocess_params']['spect_params'].get('win_length', 1200)
    n_fft = config['preprocess_params']['spect_params'].get('n_fft', 2048)
    norm_f0 = config['model_params'].get('norm_f0', True)
    frame_rate = sr // hop_length

    train_dataloader = build_dataloader(batch_size=batch_size,
                                        num_workers=4,
                                        rank=accelerator.local_process_index,
                                        world_size=accelerator.num_processes,
                                        prefetch_factor=8,
                                        )


    with accelerator.main_process_first():
        pitch_extractor = load_F0_models(config['F0_path']).to(device)

        # load model and processor
        w2v_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")
        w2v_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft").to(device)
        w2v_model.eval()

        speaker_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained("nvidia/speakerverification_en_titanet_large")
        speaker_model = speaker_model.to(device)
        speaker_model.eval()
    scheduler_params = {
        "warmup_steps": 200,
        "base_lr": 0.0001,
    }

    model_params = recursive_munch(config['model_params'])
    model = build_model(model_params)

    for k in model:
        model[k] = accelerator.prepare(model[k])

    _ = [model[key].to(device) for key in model]

    # initialize optimizers after preparing models for compatibility with FSDP
    optimizer = build_optimizer({key: model[key] for key in model},
                                scheduler_params_dict={key: scheduler_params.copy() for key in model},
                                lr=float(scheduler_params['base_lr']))

    for k, v in optimizer.optimizers.items():
        optimizer.optimizers[k] = accelerator.prepare(optimizer.optimizers[k])
        optimizer.schedulers[k] = accelerator.prepare(optimizer.schedulers[k])

    # find latest checkpoint with name pattern of 'T2V_epoch_*_step_*.pth'
    available_checkpoints = glob.glob(osp.join(log_dir, "FAcodec_epoch_*_step_*.pth"))
    if len(available_checkpoints) > 0:
        # find the checkpoint that has the highest step number
        latest_checkpoint = max(
            available_checkpoints, key=lambda x: int(x.split("_")[-1].split(".")[0])
        )
        earliest_checkpoint = min(
            available_checkpoints, key=lambda x: int(x.split("_")[-1].split(".")[0])
        )
        # delete the earliest checkpoint
        if (
            earliest_checkpoint != latest_checkpoint
            and accelerator.is_main_process
            and len(available_checkpoints) > 4
        ):
            os.remove(earliest_checkpoint)
            print(f"Removed {earliest_checkpoint}")
    else:
        latest_checkpoint = config.get("pretrained_model", "")

    with accelerator.main_process_first():
        if latest_checkpoint != '':
            model, optimizer, start_epoch, iters = load_checkpoint(model, optimizer, latest_checkpoint,
                  load_only_params=config.get('load_only_params', True), ignore_modules=[], is_distributed=accelerator.num_processes > 1)
        else:
            start_epoch = 0
            iters = 0

    content_criterion = FocalLoss(gamma=2).to(device)
    stft_criterion = MultiScaleSTFTLoss().to(device)
    mel_criterion = MelSpectrogramLoss(
        n_mels=[5, 10, 20, 40, 80, 160, 320],
        window_lengths=[32, 64, 128, 256, 512, 1024, 2048],
        mel_fmin=[0, 0, 0, 0, 0, 0, 0],
        mel_fmax=[None, None, None, None, None, None, None],
        pow=1.0,
        mag_weight=0.0,
        clamp_eps=1e-5,
    ).to(device)
    l1_criterion = L1Loss().to(device)

    for epoch in range(start_epoch, epochs):
        start_time = time.time()
        # train_dataloader.set_epoch(epoch)
        _ = [model[key].train() for key in model]
        last_time = time.time()
        for i, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            # torch.save(batch, f"latest_batch_{device}.pt")
            # train time count start
            train_start_time = time.time()

            batch = [b.to(device, non_blocking=True) for b in batch]
            waves, mels, wave_lengths, mel_input_length, noises, noise_added_flags, recon_noisy_flags = batch

            # extract semantic latent with w2v model
            waves_16k = torchaudio.functional.resample(waves, 24000, 16000)
            w2v_input = w2v_processor(waves_16k, sampling_rate=16000, return_tensors="pt").input_values.to(device)
            with torch.no_grad():
                w2v_outputs = w2v_model(w2v_input.squeeze(0)).logits
                predicted_ids = torch.argmax(w2v_outputs, dim=-1)
                phone_ids = F.interpolate(predicted_ids.unsqueeze(0).float(), mels.size(-1), mode='nearest').long().squeeze(0)

            # get clips
            mel_seg_len = min([int(mel_input_length.min().item()), max_frame_len])


            gt_mel_seg = []
            wav_seg = []
            w2v_seg = []

            for bib in range(len(mel_input_length)):
                mel_length = int(mel_input_length[bib].item())

                random_start = np.random.randint(0, mel_length - mel_seg_len) if mel_length != mel_seg_len else 0
                gt_mel_seg.append(mels[bib, :, random_start:random_start + mel_seg_len])

                # w2v_seg.append(w2v_latent[bib, :, random_start:random_start + mel_seg_len])
                w2v_seg.append(phone_ids[bib, random_start:random_start + mel_seg_len])

                y = waves[bib][random_start * 300:(random_start + mel_seg_len) * 300]

                wav_seg.append(y.to(device))

            gt_mel_seg = torch.stack(gt_mel_seg).detach()

            wav_seg = torch.stack(wav_seg).float().detach().unsqueeze(1)
            w2v_seg = torch.stack(w2v_seg).float().detach()

            with torch.no_grad():
                real_norm = log_norm(gt_mel_seg.unsqueeze(1)).squeeze(1).detach()
                F0_real, _, _ = pitch_extractor(gt_mel_seg.unsqueeze(1))

            # normalize f0
            # Remove unvoiced frames (replace with -1)
            gt_glob_f0s = []
            if not norm_f0:
                f0_targets = F0_real
            else:
                f0_targets = []
                for bib in range(len(F0_real)):
                    voiced_indices = F0_real[bib] > 5.0
                    f0_voiced = F0_real[bib][voiced_indices]

                    if len(f0_voiced) != 0:
                        # Convert to log scale
                        log_f0 = f0_voiced.log2()

                        # Calculate mean and standard deviation
                        mean_f0 = log_f0.mean()
                        std_f0 = log_f0.std()

                        # Normalize the F0 sequence
                        normalized_f0 = (log_f0 - mean_f0) / std_f0

                        # Create the normalized F0 sequence with unvoiced frames
                        normalized_sequence = torch.zeros_like(F0_real[bib])
                        normalized_sequence[voiced_indices] = normalized_f0
                        normalized_sequence[~voiced_indices] = -10  # Assign -10 to unvoiced frames

                        gt_glob_f0s.append(mean_f0)
                    else:
                        normalized_sequence = torch.zeros_like(F0_real[bib]) - 10.0
                        gt_glob_f0s.append(torch.tensor(0.0).to(device))

                    # f0_targets.append(normalized_sequence[single_side_context // 200:-single_side_context // 200])
                    f0_targets.append(normalized_sequence)
                f0_targets = torch.stack(f0_targets).to(device)
                # fill nan with -10
                f0_targets[torch.isnan(f0_targets)] = -10.0
                # fill inf with -10
                f0_targets[torch.isinf(f0_targets)] = -10.0
            # if frame_rate not equal to 80, interpolate f0 from frame rate of 80 to target frame rate
            if frame_rate != 80:
                f0_targets = F.interpolate(f0_targets.unsqueeze(1), mel_seg_len // 80 * frame_rate, mode='nearest').squeeze(1)
                w2v_seg = F.interpolate(w2v_seg, mel_seg_len // 80 * frame_rate, mode='nearest')

            # add noise according to noise_added_flags
            # randomly determine a batch of signal to noise ratios, ranges in [1, 5]
            snr = torch.rand(wav_seg.size(0)).to(device) * 4 + 1
            # scale noise according to snr
            speech_energy = torch.sum(wav_seg ** 2, dim=-1).squeeze(-1)
            noise_energy = torch.sum(noises ** 2, dim=-1)

            target_noise_energy = speech_energy / snr

            scale_factor = torch.sqrt(target_noise_energy / (noise_energy + 1e-8))
            scaled_noises = noises * scale_factor.unsqueeze(1) * noise_added_flags.unsqueeze(1)

            wav_seg_input = wav_seg + scaled_noises.unsqueeze(1)
            wav_seg_target = wav_seg + scaled_noises.unsqueeze(1) * recon_noisy_flags[:, None, None]

            z = model.encoder(wav_seg_input)
            z, quantized, commitment_loss, codebook_loss, timbre = model.quantizer(z, wav_seg_input,
                                                                                   noise_added_flags,
                                                                                   recon_noisy_flags,
                                                                                   n_c=2,
                                                                                   full_waves=waves,
                                                                                   wave_lens=wave_lengths)
            preds, rev_preds = model.fa_predictors(quantized, timbre)

            pred_wave = model.decoder(z)

            len_diff = wav_seg_target.size(-1) - pred_wave.size(-1)
            if len_diff > 0:
                wav_seg_target = wav_seg_target[..., len_diff // 2:-len_diff // 2]


            # discriminator loss
            d_fake = model.discriminator(pred_wave.detach())
            d_real = model.discriminator(wav_seg_target)
            loss_d = 0
            for x_fake, x_real in zip(d_fake, d_real):
                loss_d += torch.mean(x_fake[-1] ** 2)
                loss_d += torch.mean((1 - x_real[-1]) ** 2)


            optimizer.zero_grad()
            accelerator.backward(loss_d)
            grad_norm_d = torch.nn.utils.clip_grad_norm_(model.discriminator.parameters(), 10.0)
            optimizer.step('discriminator')
            optimizer.scheduler(key='discriminator')

            # generator loss
            signal = AudioSignal(wav_seg_target, sample_rate=24000)
            recons = AudioSignal(pred_wave, sample_rate=24000)
            stft_loss = stft_criterion(recons, signal)
            mel_loss = mel_criterion(recons, signal)
            waveform_loss = l1_criterion(recons, signal)

            d_fake = model.discriminator(pred_wave)
            d_real = model.discriminator(wav_seg_target)

            loss_g = 0
            for x_fake in d_fake:
                loss_g += torch.mean((1 - x_fake[-1]) ** 2)

            loss_feature = 0

            for i in range(len(d_fake)):
                for j in range(len(d_fake[i]) - 1):
                    loss_feature += F.l1_loss(d_fake[i][j], d_real[i][j].detach())

            pred_f0, pred_uv = preds['f0'], preds['uv']
            rev_pred_f0, rev_pred_uv = rev_preds['rev_f0'], rev_preds['rev_uv']

            common_min_size = min(pred_f0.size(-2), f0_targets.size(-1))
            f0_targets = f0_targets[..., :common_min_size]
            real_norm = real_norm[..., :common_min_size]

            f0_loss = F.smooth_l1_loss(f0_targets, pred_f0.squeeze(-1)[..., :common_min_size])
            uv_loss = F.smooth_l1_loss(real_norm, pred_uv.squeeze(-1)[..., :common_min_size])
            rev_f0_loss = F.smooth_l1_loss(f0_targets, rev_pred_f0.squeeze(-1)[..., :common_min_size]) if rev_pred_f0 is not None else torch.FloatTensor([0]).to(device)
            rev_uv_loss = F.smooth_l1_loss(real_norm, rev_pred_uv.squeeze(-1)[..., :common_min_size]) if rev_pred_uv is not None else torch.FloatTensor([0]).to(device)

            tot_f0_loss = f0_loss + rev_f0_loss
            tot_uv_loss = uv_loss + rev_uv_loss

            pred_content = preds['content']
            rev_pred_content = rev_preds['rev_content']

            target_content_latents = w2v_seg[..., :common_min_size]

            content_loss = content_criterion(pred_content.transpose(1, 2)[..., :common_min_size], target_content_latents.long())
            rev_content_loss = content_criterion(rev_pred_content.transpose(1, 2)[..., :common_min_size], target_content_latents.long()) \
                if rev_pred_content is not None else torch.FloatTensor([0]).to(device)

            tot_content_loss = content_loss + rev_content_loss

            spk_logits = torch.cat([speaker_model.infer_segment(w16.cpu()[..., :wl])[1] for w16, wl in zip(waves_16k, wave_lengths)], dim=0)
            spk_labels = spk_logits.argmax(dim=-1)

            spk_pred_logits = preds['timbre']
            spk_loss = F.cross_entropy(spk_pred_logits, spk_labels)
            x_spk_pred_logits = rev_preds['x_timbre']

            x_spk_loss = F.cross_entropy(x_spk_pred_logits, spk_labels) if x_spk_pred_logits is not None else torch.FloatTensor([0]).to(device)

            tot_spk_loss = spk_loss + x_spk_loss

            # global f0 loss
            # get average f0
            gt_glob_f0s = torch.stack(gt_glob_f0s)
            global_f0_loss = F.smooth_l1_loss(gt_glob_f0s.unsqueeze(1), preds['global_f0'])
            rev_global_f0_loss = F.smooth_l1_loss(gt_glob_f0s.unsqueeze(1), rev_preds['rev_global_f0']) if rev_preds['rev_global_f0'] is not None else torch.FloatTensor([0]).to(device)

            loss_gen_all = mel_loss * 15.0 + loss_feature * 1.0 + loss_g * 1.0 + commitment_loss * 0.25 + codebook_loss * 1.0 \
                            + tot_f0_loss * 1.0 + tot_uv_loss * 1.0 + tot_content_loss * 5.0 + tot_spk_loss * 5.0 + global_f0_loss * 1.0 + rev_global_f0_loss * 1.0

            optimizer.zero_grad()
            accelerator.backward(loss_gen_all)
            grad_norm_g = torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), 1000.0)
            grad_norm_g2 = torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), 1000.0)
            grad_norm_g3 = torch.nn.utils.clip_grad_norm_(model.quantizer.parameters(), 1000.0)
            grad_norm_g4 = torch.nn.utils.clip_grad_norm_(model.fa_predictors.parameters(), 1000.0)

            optimizer.step('encoder')
            optimizer.step('decoder')
            optimizer.step('quantizer')
            optimizer.step('fa_predictors')

            optimizer.scheduler(key='encoder')
            optimizer.scheduler(key='decoder')
            optimizer.scheduler(key='quantizer')
            optimizer.scheduler(key='fa_predictors')


            # optimizer.step()
            # train time count end
            train_time_per_step = time.time() - train_start_time

            if iters % log_interval == 0 and accelerator.is_main_process:
                with torch.no_grad():
                    cur_lr = optimizer.schedulers['encoder'].get_last_lr()[0] if i != 0 else 0
                    # log print and tensorboard
                    print("Epoch %d, Iteration %d, Gen Loss: %.4f, Disc Loss: %.4f, mel Loss: %.4f, Time: %.4f" % (
                        epoch, iters, loss_gen_all.item(), loss_d.item(), mel_loss.item(), train_time_per_step))
                    print("f0 Loss: %.4f, uv Loss: %.4f, content Loss: %.4f, spk Loss: %.4f, global_f0_loss: %.4f" % (
                        f0_loss.item(), uv_loss.item(), content_loss.item(), spk_loss.item(), global_f0_loss.item()))
                    print("rev f0 Loss: %.4f, rev uv Loss: %.4f, rev content Loss: %.4f, x spk Loss: %.4f, rev global f0 Loss: %.4f" % (
                        rev_f0_loss.item(), rev_uv_loss.item(), rev_content_loss.item(), x_spk_loss.item(), rev_global_f0_loss.item())
                    )
                    writer.add_scalar('train/lr', cur_lr, iters)
                    writer.add_scalar('train/time', train_time_per_step, iters)

                    writer.add_scalar('grad_norm/encoder', grad_norm_g, iters)
                    writer.add_scalar('grad_norm/decoder', grad_norm_g2, iters)
                    writer.add_scalar('grad_norm/fa_quantizer', grad_norm_g3, iters)
                    writer.add_scalar('grad_norm/fa_predictors', grad_norm_g4, iters)

                    writer.add_scalar('train/loss_gen_all', loss_gen_all.item(), iters)
                    writer.add_scalar('train/loss_disc_all', loss_d.item(), iters)
                    writer.add_scalar('train/wav_loss', waveform_loss.item(), iters)
                    writer.add_scalar('train/mel_loss', mel_loss.item(), iters)
                    writer.add_scalar('train/stft_loss', stft_loss.item(), iters)
                    writer.add_scalar('train/feat_loss', loss_feature.item(), iters)

                    writer.add_scalar('train/commit_loss', commitment_loss.item(), iters)
                    writer.add_scalar('train/codebook_loss', codebook_loss.item(), iters)

                    writer.add_scalar('pred/f0_loss', f0_loss.item(), iters)
                    writer.add_scalar('pred/uv_loss', uv_loss.item(), iters)
                    writer.add_scalar('pred/content_loss', content_loss.item(), iters)
                    writer.add_scalar('pred/spk_loss', spk_loss.item(), iters)
                    writer.add_scalar('pred/global_f0_loss', global_f0_loss.item(), iters)

                    writer.add_scalar('rev_pred/rev_f0_loss', rev_f0_loss.item(), iters)
                    writer.add_scalar('rev_pred/rev_uv_loss', rev_uv_loss.item(), iters)
                    writer.add_scalar('rev_pred/rev_content_loss', rev_content_loss.item(), iters)
                    writer.add_scalar('rev_pred/x_spk_loss', x_spk_loss.item(), iters)
                    writer.add_scalar('rev_pred/rev_global_f0_loss', rev_global_f0_loss.item(), iters)


                    print('Time elasped:', time.time() - start_time)
            if iters % (log_interval * 10) == 0 and accelerator.is_main_process:
                with torch.no_grad():
                    writer.add_audio('train/gt_audio', wav_seg_input[0], iters, sample_rate=24000)
                    writer.add_audio('train/pred_audio', pred_wave[0], iters, sample_rate=24000)

            if iters % (log_interval * 1000) == 0 and accelerator.is_main_process:
                with torch.no_grad():
                    # put ground truth audio
                    writer.add_audio('full/gt_audio', waves[0], iters, sample_rate=16000)

                    # without timbre norm
                    z = model.encoder(waves[0, :wave_lengths[0]][None, None, ...].to(device).float())
                    z, quantized, commitment_loss, codebook_loss, timbre = model.quantizer(z, waves[0, :wave_lengths[0]][None, None, ...],
                         torch.zeros(1).to(device).bool(), torch.zeros(1).to(device).bool())

                    z2 = model.encoder(waves[1, :wave_lengths[1]][None, None, ...].to(device).float())
                    z2, quantized2, commitment_loss2, codebook_loss2, timbre2 = model.quantizer(z2, waves[1, :wave_lengths[1]][None, None, ...],
                         torch.zeros(1).to(device).bool(), torch.zeros(1).to(device).bool())

                    p_pred_wave = model.decoder(quantized[0])
                    c_pred_wave = model.decoder(quantized[1])
                    r_pred_wave = model.decoder(quantized[2])
                    pc_pred_wave = model.decoder(quantized[0] + quantized[1])
                    pr_pred_wave = model.decoder(quantized[0] + quantized[2])
                    pcr_pred_wave = model.decoder(quantized[0] + quantized[1] + quantized[2])
                    full_pred_wave = model.decoder(z)
                    x = quantized[0] + quantized[1] + quantized[2]
                    style2 = model.quantizer.module.timbre_linear(timbre2).unsqueeze(2)  # (B, 2d, 1)
                    gamma, beta = style2.chunk(2, 1)  # (B, d, 1)
                    x = x.transpose(1, 2)
                    x = model.quantizer.module.timbre_norm(x)
                    x = x.transpose(1, 2)
                    x = x * gamma + beta
                    vc_pred_wave = model.decoder(x)
                    writer.add_audio('partial/pred_audio_p', p_pred_wave[0], iters, sample_rate=sr)
                    writer.add_audio('partial/pred_audio_c', c_pred_wave[0], iters, sample_rate=sr)
                    writer.add_audio('partial/pred_audio_r', r_pred_wave[0], iters, sample_rate=sr)
                    writer.add_audio('partial/pred_audio_pc', pc_pred_wave[0], iters, sample_rate=sr)
                    writer.add_audio('partial/pred_audio_pr', pr_pred_wave[0], iters, sample_rate=sr)
                    writer.add_audio('partial/pred_audio_pcr', pcr_pred_wave[0], iters, sample_rate=sr)
                    writer.add_audio('full/pred_audio', full_pred_wave[0], iters, sample_rate=sr)

                    writer.add_audio('vc/ref_audio', waves[1], iters, sample_rate=sr)
                    writer.add_audio('vc/pred_audio', vc_pred_wave[0], iters, sample_rate=sr)
            if iters % save_interval == 0 and accelerator.is_main_process:
                print('Saving..')
                state = {
                    'net': {key: model[key].state_dict() for key in model},
                    'optimizer': optimizer.state_dict(),
                    'scheduler': optimizer.scheduler_state_dict(),
                    'iters': iters,
                    'epoch': epoch,
                }
                save_path = osp.join(log_dir, 'FAcodec_epoch_%05d_step_%05d.pth' % (epoch, iters))
                torch.save(state, save_path)

                # find all checkpoints and remove old ones
                checkpoints = glob.glob(osp.join(log_dir, 'FAcodec_epoch_*.pth'))
                if len(checkpoints) > 5:
                    # sort by step
                    checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
                    # remove all except last 5
                    for cp in checkpoints[:-5]:
                        os.remove(cp)
            iters = iters + 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='configs/config.yml')
    args = parser.parse_args()
    main(args)