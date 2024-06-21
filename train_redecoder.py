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

from stream_dataset_tencent import build_dataloader
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
# from speechtokenizer import SpeechTokenizer

from dac.nn.loss import MultiScaleSTFTLoss, MelSpectrogramLoss, GANLoss, L1Loss
from audiotools import AudioSignal

from transformers import SeamlessM4TFeatureExtractor, Wav2Vec2BertModel
import glob
# import nemo.collections.asr as nemo_asr


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
    device = accelerator.device# if accelerator.num_processes > 1 else torch.device('cpu')

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

    model_params = recursive_munch(config['model_params'])
    with accelerator.main_process_first():
        codec_encoder = build_model(model_params, stage='encoder')
        codec_encoder, _, _, _ = load_checkpoint(codec_encoder, None, config['pretrained_encoder'],
                                                               load_only_params=True,
                                                               ignore_modules=[],
                                                               is_distributed=False)
        _ = [codec_encoder[key].eval() for key in codec_encoder]
        _ = [codec_encoder[key].to(device) for key in codec_encoder]
    scheduler_params = {
        "warmup_steps": 200,
        "base_lr": 0.0001,
    }


    model = build_model(model_params, stage='redecoder')
    is_timbre_norm = model_params.timbre_norm

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
    available_checkpoints = glob.glob(osp.join(log_dir, "FAredecoder_epoch_*_step_*.pth"))
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
        train_dataloader.sampler.set_epoch(epoch)
        _ = [model[key].train() for key in model]
        last_time = time.time()
        for i, batch in enumerate(train_dataloader):
        # for i in range(5):
            optimizer.zero_grad()
            # torch.save(batch, f"latest_batch_{device}.pt")
            # train time count start
            train_start_time = time.time()

            batch = [b.to(device, non_blocking=True) for b in batch]
            waves, mels, wave_lengths, mel_input_length, noises, noise_added_flags, recon_noisy_flags = batch
            # waves = torch.randn(4, 24000 * 10).to(device)
            # wave_lengths = torch.tensor([24000 * 10] * 4).to(device)
            # mels = torch.randn(4, 80, 80*10).to(device)
            # mel_input_length = torch.tensor([80*10] * 4).to(device)
            # print(waves.shape)
            # print(f"dataloader takes {time.time() - last_time}")
            # last_time = time.time()
            # continue
            # with torch.no_grad():
            #     z = codec_encoder.encoder(waves)
            #     z, quantized, commitment_loss, codebook_loss, timbre, codes = codec_encoder.quantizer(z, waves,
            #                                                                            torch.ones(waves.size(0)).to(device).bool(),
            #                                                                            torch.ones(waves.size(0)).to(device).bool(),
            #                                                                            n_c=2,
            #                                                                            full_waves=waves,
            #                                                                            wave_lens=wave_lengths,
            #                                                                            return_codes=True)

            if model_params.encoder_type == 'wavenet':
                # get clips
                mel_seg_len = min([int(mel_input_length.min().item()), max_frame_len])


                gt_mel_seg = []
                wav_seg = []

                for bib in range(len(mel_input_length)):
                    mel_length = int(mel_input_length[bib].item())

                    random_start = np.random.randint(0, mel_length - mel_seg_len) if mel_length != mel_seg_len else 0
                    gt_mel_seg.append(mels[bib, :, random_start:random_start + mel_seg_len])

                    y = waves[bib][random_start * 300:(random_start + mel_seg_len) * 300]

                    wav_seg.append(y.to(device))


                # en = [torch.stack(e) for e in en]
                gt_mel_seg = torch.stack(gt_mel_seg).detach()

                wav_seg = torch.stack(wav_seg).float().detach().unsqueeze(1)
                with torch.no_grad():
                    z = codec_encoder.encoder(wav_seg)
                    z, quantized, commitment_loss, codebook_loss, timbre, codes = codec_encoder.quantizer(z, wav_seg,
                                       torch.ones(waves.size(0)).to(device).bool(),
                                       torch.ones(waves.size(0)).to(device).bool(),
                                       n_c=2,
                                       full_waves=waves,
                                       wave_lens=wave_lengths,
                                       return_codes=True)
                encoder_out = model.encoder(codes[0], codes[1], timbre)
            elif model_params.encoder_type == 'mamba':
                with torch.no_grad():
                    waves_input = waves.unsqueeze(1)
                    z = codec_encoder.encoder(waves_input)
                    z, quantized, commitment_loss, codebook_loss, timbre, codes = codec_encoder.quantizer(z,
                                  waves_input,
                                  torch.ones(waves_input.size(0)).to(device).bool(),
                                  torch.ones(waves_input.size(0)).to(device).bool(),
                                  n_c=2,
                                  full_waves=waves_input,
                                  wave_lens=wave_lengths,
                                  return_codes=True)
                encoder_out = model.encoder(codes[0], codes[1], z, timbre)
                encoder_out = encoder_out[..., 0::2]

                # get clips
                mel_seg_len = min([int(mel_input_length.min().item()), max_frame_len])

                out_seg = []
                wav_seg = []

                for bib in range(len(mel_input_length)):
                    mel_length = int(mel_input_length[bib].item())

                    random_start = np.random.randint(0, mel_length - mel_seg_len) if mel_length != mel_seg_len else 0
                    out_seg.append(encoder_out[bib, :, random_start:random_start + mel_seg_len])

                    y = waves[bib][random_start * 300:(random_start + mel_seg_len) * 300]

                    wav_seg.append(y.to(device))

                # en = [torch.stack(e) for e in en]
                encoder_out = torch.stack(out_seg)
                wav_seg = torch.stack(wav_seg).float().unsqueeze(1)
            else:
                raise NotImplementedError


            pred_wave = model.decoder(encoder_out)

            len_diff = wav_seg.size(-1) - pred_wave.size(-1)
            if len_diff > 0:
                wav_seg = wav_seg[..., len_diff // 2:-len_diff // 2]


            # discriminator loss
            d_fake = model.discriminator(pred_wave.detach())
            d_real = model.discriminator(wav_seg)
            loss_d = 0
            for x_fake, x_real in zip(d_fake, d_real):
                loss_d += torch.mean(x_fake[-1] ** 2)
                loss_d += torch.mean((1 - x_real[-1]) ** 2)

            # # reverse timbre predictor loss
            # x_spk_pred = model.rev_timbre_predictor(quantized[0].detach() + quantized[1].detach() + quantized[2].detach())[0]
            # x_spk_loss_d = spk_criterion(x_spk_pred, spk_embedding, torch.ones(spk_embedding.size(0)).to(device))


            optimizer.zero_grad()
            accelerator.backward(loss_d)
            grad_norm_d = torch.nn.utils.clip_grad_norm_(model.discriminator.parameters(), 10.0)
            optimizer.step('discriminator')
            optimizer.scheduler(key='discriminator')

            # accelerator.backward(x_spk_loss_d)
            # grad_norm_rev_spk = torch.nn.utils.clip_grad_norm_(model.rev_timbre_predictor.parameters(), 10.0)
            # optimizer.step('rev_timbre_predictor')
            # optimizer.scheduler(key='rev_timbre_predictor')

            # generator loss
            signal = AudioSignal(wav_seg, sample_rate=24000)
            recons = AudioSignal(pred_wave, sample_rate=24000)
            stft_loss = stft_criterion(recons, signal)
            mel_loss = mel_criterion(recons, signal)
            waveform_loss = l1_criterion(recons, signal)

            d_fake = model.discriminator(pred_wave)
            d_real = model.discriminator(wav_seg)

            loss_g = 0
            for x_fake in d_fake:
                loss_g += torch.mean((1 - x_fake[-1]) ** 2)

            loss_feature = 0

            for i in range(len(d_fake)):
                for j in range(len(d_fake[i]) - 1):
                    loss_feature += F.l1_loss(d_fake[i][j], d_real[i][j].detach())

            loss_gen_all = mel_loss * 15.0 + loss_feature * 1.0 + loss_g * 1.0

            optimizer.zero_grad()
            accelerator.backward(loss_gen_all)
            grad_norm_g = torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), 1000.0)
            grad_norm_g2 = torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), 1000.0)

            optimizer.step('encoder')
            optimizer.step('decoder')

            optimizer.scheduler(key='encoder')
            optimizer.scheduler(key='decoder')


            # optimizer.step()
            # train time count end
            train_time_per_step = time.time() - train_start_time

            if iters % log_interval == 0 and accelerator.is_main_process:
                with torch.no_grad():
                    cur_lr = optimizer.schedulers['encoder'].get_last_lr()[0] if i != 0 else 0
                    # log print and tensorboard
                    print("Epoch %d, Iteration %d, Gen Loss: %.4f, Disc Loss: %.4f, mel Loss: %.4f, Time: %.4f" % (
                        epoch, iters, loss_gen_all.item(), loss_d.item(), mel_loss.item(), train_time_per_step))
                    writer.add_scalar('train/lr', cur_lr, iters)
                    writer.add_scalar('train/time', train_time_per_step, iters)

                    writer.add_scalar('grad_norm/encoder', grad_norm_g, iters)
                    writer.add_scalar('grad_norm/decoder', grad_norm_g2, iters)

                    writer.add_scalar('train/loss_gen_all', loss_gen_all.item(), iters)
                    writer.add_scalar('train/loss_disc_all', loss_d.item(), iters)
                    writer.add_scalar('train/wav_loss', waveform_loss.item(), iters)
                    writer.add_scalar('train/mel_loss', mel_loss.item(), iters)
                    writer.add_scalar('train/stft_loss', stft_loss.item(), iters)
                    writer.add_scalar('train/feat_loss', loss_feature.item(), iters)


                    print('Time elasped:', time.time() - start_time)
            if iters % (log_interval * 10) == 0 and accelerator.is_main_process:
                with torch.no_grad():
                    writer.add_audio('train/gt_audio', wav_seg[0], iters, sample_rate=24000)
                    writer.add_audio('train/pred_audio', pred_wave[0], iters, sample_rate=24000)

            if iters % (log_interval * 1000) == 0 and accelerator.is_main_process:
                with torch.no_grad():
                    # put ground truth audio
                    writer.add_audio('full/gt_audio', waves[0], iters, sample_rate=24000)

                    # # without timbre norm
                    # z = model.encoder(waves[0, :wave_lengths[0]][None, None, ...].to(device).float())
                    # if is_timbre_norm:
                    #     z, quantized, commitment_loss, codebook_loss, timbre = model.quantizer(z, waves[0, :wave_lengths[0]][None, None, ...],
                    #          torch.zeros(1).to(device).bool(), torch.zeros(1).to(device).bool())
                    # else:
                    #     z, quantized, commitment_loss, codebook_loss = model.quantizer(z, wav_seg_input,
                    #           torch.zeros(1).to(device).bool(), torch.zeros(1).to(device).bool())
                    #
                    # z2 = model.encoder(waves[1, :wave_lengths[1]][None, None, ...].to(device).float())
                    # if is_timbre_norm:
                    #     z2, quantized2, commitment_loss2, codebook_loss2, timbre2 = model.quantizer(z2, waves[1, :wave_lengths[1]][None, None, ...],
                    #          torch.zeros(1).to(device).bool(), torch.zeros(1).to(device).bool())
                    # else:
                    #     z2, quantized, commitment_loss, codebook_loss = model.quantizer(z2, wav_seg_input,
                    #           torch.zeros(1).to(device).bool(), torch.zeros(1).to(device).bool())
                    #
                    # if is_timbre_norm:
                    #     p_pred_wave = model.decoder(quantized[0])
                    #     c_pred_wave = model.decoder(quantized[1])
                    #     r_pred_wave = model.decoder(quantized[2])
                    #     pc_pred_wave = model.decoder(quantized[0] + quantized[1])
                    #     pr_pred_wave = model.decoder(quantized[0] + quantized[2])
                    #     pcr_pred_wave = model.decoder(quantized[0] + quantized[1] + quantized[2])
                    #     full_pred_wave = model.decoder(z)
                    #     x = quantized[0] + quantized[1] + quantized[2]
                    #     style2 = model.quantizer.module.timbre_linear(timbre2).unsqueeze(2)  # (B, 2d, 1)
                    #     gamma, beta = style2.chunk(2, 1)  # (B, d, 1)
                    #     x = x.transpose(1, 2)
                    #     x = model.quantizer.module.timbre_norm(x)
                    #     x = x.transpose(1, 2)
                    #     x = x * gamma + beta
                    #     vc_pred_wave = model.decoder(x)
                    #     writer.add_audio('partial/pred_audio_p', p_pred_wave[0], iters, sample_rate=sr)
                    #     writer.add_audio('partial/pred_audio_c', c_pred_wave[0], iters, sample_rate=sr)
                    #     writer.add_audio('partial/pred_audio_r', r_pred_wave[0], iters, sample_rate=sr)
                    #     writer.add_audio('partial/pred_audio_pc', pc_pred_wave[0], iters, sample_rate=sr)
                    #     writer.add_audio('partial/pred_audio_pr', pr_pred_wave[0], iters, sample_rate=sr)
                    #     writer.add_audio('partial/pred_audio_pcr', pcr_pred_wave[0], iters, sample_rate=sr)
                    #     writer.add_audio('full/pred_audio', full_pred_wave[0], iters, sample_rate=sr)
                    #
                    #     writer.add_audio('vc/ref_audio', waves[1], iters, sample_rate=sr)
                    #     writer.add_audio('vc/pred_audio', vc_pred_wave[0], iters, sample_rate=sr)
                    # else:
                    #     p_pred_wave = model.decoder(quantized[0])
                    #     c_pred_wave = model.decoder(quantized[1])
                    #     t_pred_wave = model.decoder(quantized[2])
                    #     r_pred_wave = model.decoder(quantized[3])
                    #     pc_pred_wave = model.decoder(quantized[0] + quantized[1])
                    #     pt_pred_wave = model.decoder(quantized[0] + quantized[2])
                    #     ct_pred_wave = model.decoder(quantized[1] + quantized[2])
                    #     pct_pred_wave = model.decoder(quantized[0] + quantized[1] + quantized[2])
                    #     full_pred_wave = model.decoder(quantized[0] + quantized[1] + quantized[2] + quantized[3])
                    #
                    #     writer.add_audio('partial/pred_audio_p', p_pred_wave[0], iters, sample_rate=sr)
                    #     writer.add_audio('partial/pred_audio_c', c_pred_wave[0], iters, sample_rate=sr)
                    #     writer.add_audio('partial/pred_audio_t', t_pred_wave[0], iters, sample_rate=sr)
                    #     writer.add_audio('partial/pred_audio_r', r_pred_wave[0], iters, sample_rate=sr)
                    #     writer.add_audio('partial/pred_audio_pc', pc_pred_wave[0], iters, sample_rate=sr)
                    #     writer.add_audio('partial/pred_audio_pt', pt_pred_wave[0], iters, sample_rate=sr)
                    #     writer.add_audio('partial/pred_audio_ct', ct_pred_wave[0], iters, sample_rate=sr)
                    #     writer.add_audio('partial/pred_audio_pct', pct_pred_wave[0], iters, sample_rate=sr)
                    #     writer.add_audio('full/pred_audio', full_pred_wave[0], iters, sample_rate=sr)
            if iters % save_interval == 0 and accelerator.is_main_process:
                print('Saving..')
                state = {
                    'net': {key: model[key].state_dict() for key in model},
                    'optimizer': optimizer.state_dict(),
                    'scheduler': optimizer.scheduler_state_dict(),
                    'iters': iters,
                    'epoch': epoch,
                }
                save_path = osp.join(log_dir, 'FAredecoder_epoch_%05d_step_%05d.pth' % (epoch, iters))
                torch.save(state, save_path)

                # find all checkpoints and remove old ones
                checkpoints = glob.glob(osp.join(log_dir, 'FAredecoder_epoch_*.pth'))
                if len(checkpoints) > 5:
                    # sort by step
                    checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
                    # remove all except last 5
                    for cp in checkpoints[:-5]:
                        os.remove(cp)
            iters = iters + 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='configs/config_redecoder_v2.yml')
    args = parser.parse_args()
    main(args)