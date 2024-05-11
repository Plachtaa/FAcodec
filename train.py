import shutil
import click
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
from torchmetrics.classification import MulticlassAccuracy

import logging
from accelerate.logging import get_logger
from speechtokenizer import SpeechTokenizer

from dac.nn.loss import MultiScaleSTFTLoss, MelSpectrogramLoss, GANLoss, L1Loss
from audiotools import AudioSignal


logger = get_logger(__name__, log_level="DEBUG")
# torch.autograd.set_detect_anomaly(True)

def main(args):
    config_path = args.config_path
    config = yaml.safe_load(open(config_path))

    log_dir = config['log_dir']
    if not osp.exists(log_dir): os.makedirs(log_dir, exist_ok=True)
    shutil.copy(config_path, osp.join(log_dir, osp.basename(config_path)))
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True, broadcast_buffers=False)
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
    device = accelerator.device

    epochs = config.get('epochs', 200)
    log_interval = config.get('log_interval', 10)
    saving_epoch = config.get('save_freq', 2)
    save_interval = config.get('save_interval', 1000)

    data_params = config.get('data_params', None)
    sr = config['preprocess_params'].get('sr', 16000)
    train_path = data_params['train_data']
    val_path = data_params['val_data']
    root_path = data_params['root_path']
    min_length = data_params['min_length']
    max_frame_len = config.get('max_len', 80)
    discriminator_iter_start = config['loss_params'].get('discriminator_iter_start', 0)
    loss_params = config.get('loss_params', {})

    train_dataloader = build_dataloader(train_path,
                                        root_path,
                                        min_length=min_length,
                                        batch_size=batch_size,
                                        batch_length=None,
                                        num_workers=4,
                                        rank=accelerator.local_process_index,
                                        world_size=accelerator.num_processes,
                                        dataset_config={},
                                        device=device,
                                        n_repeats=1,
                                        return_dict=True)


    with accelerator.main_process_first():
        pitch_extractor = load_F0_models(config['F0_path']).to(device)

        # load pretrained audio codec
        w2v_config_path = './w2v_models/speechtokenizer_hubert_avg_config.json'
        w2v_ckpt_path = './w2v_models/SpeechTokenizer.pt'
        w2v_model = SpeechTokenizer.load_from_checkpoint(w2v_config_path, w2v_ckpt_path).to(device)
        w2v_model.eval()

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

    with accelerator.main_process_first():
        if config.get('pretrained_model', '') != '':
            model, optimizer, start_epoch, iters = load_checkpoint(model, optimizer, config['pretrained_model'],
                  load_only_params=config.get('load_only_params', True), ignore_modules=[], is_distributed=accelerator.num_processes > 1)
        else:
            start_epoch = 0
            iters = 0
    spk_criterion = FocalLoss(gamma=2).to(device)

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

    train_dataloader = accelerator.prepare(train_dataloader)

    content_metric = MulticlassAccuracy(
            top_k=10,
            num_classes=1024,
            ignore_index=-1,
        ).to(device)




    for epoch in range(start_epoch, epochs):
        start_time = time.time()
        train_dataloader.set_epoch(epoch)
        _ = [model[key].train() for key in model]

        for i, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            # torch.save(batch, f"latest_batch_{device}.pt")
            # train time count start
            train_start_time = time.time()

            waves = batch[0]
            paths = batch[-1]
            batch = [b.to(device, non_blocking=True) if type(b) == torch.Tensor else b for b in batch[1:-1]]
            texts, input_lengths, mels, mel_input_length, speaker_labels, langs = batch

            # get clips
            mel_seg_len = min([int(mel_input_length.min().item()), max_frame_len])


            gt_mel_seg = []
            gt_mel_seg4c = []
            wav_seg = []
            wav_seg4c = [] # for content extractor, add additional context

            for bib in range(len(mel_input_length)):
                mel_length = int(mel_input_length[bib].item())

                random_start = np.random.randint(0, mel_length - mel_seg_len) if mel_length != mel_seg_len else 0
                # for q, layer_q in enumerate(quantized):
                #     en[q].append(layer_q[bib, :, random_start:random_start + mel_seg_len])
                gt_mel_seg.append(mels[bib, :, random_start:random_start + mel_seg_len])

                single_side_context = (3 * 80 - mel_seg_len) // 2
                padded_mel = F.pad(mels[bib], (160, 160), 'constant', value=-10.0)
                random_start_pad = random_start + 160
                gt_mel_seg4c.append(padded_mel[:, random_start_pad - single_side_context:random_start_pad + mel_seg_len + single_side_context])

                y = waves[bib][random_start * 200:(random_start + mel_seg_len) * 200]

                if len(y) < mel_seg_len * 200:
                    y = np.pad(y, (0, mel_seg_len * 200 - len(y)), 'constant')
                wav_seg.append(torch.from_numpy(y).to(device))

                single_side_context = (3 * 16000 - mel_seg_len * 200) // 2
                padded_wave = np.pad(waves[bib], (32000, 32000), 'constant') # additional 1 sec context on both sides
                random_start = random_start + 160
                y_c = padded_wave[random_start * 200 - single_side_context:(random_start + mel_seg_len) * 200 + single_side_context]
                wav_seg4c.append(torch.from_numpy(y_c).to(device))
                assert (y == y_c[single_side_context:-single_side_context]).prod()


            # en = [torch.stack(e) for e in en]
            gt_mel_seg = torch.stack(gt_mel_seg).detach()
            gt_mel_seg4c = torch.stack(gt_mel_seg4c).detach()

            wav_seg = torch.stack(wav_seg).float().detach().unsqueeze(1)
            wav_seg4c = torch.stack(wav_seg4c).float().detach().unsqueeze(1)

            with torch.no_grad():
                real_norm = log_norm(gt_mel_seg.unsqueeze(1)).squeeze(1).detach()
                # F0_real, _, _ = pitch_extractor(gt_mel_seg4c.unsqueeze(1))
                F0_real, _, _ = pitch_extractor(gt_mel_seg.unsqueeze(1))

            # normalize f0
            # Remove unvoiced frames (replace with -1)
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
                else:
                    normalized_sequence = torch.zeros_like(F0_real[bib]) - 10.0

                # f0_targets.append(normalized_sequence[single_side_context // 200:-single_side_context // 200])
                f0_targets.append(normalized_sequence)
            f0_targets = torch.stack(f0_targets).to(device)
            # fill nan with -10
            f0_targets[torch.isnan(f0_targets)] = -10.0
            # fill inf with -10
            f0_targets[torch.isinf(f0_targets)] = -10.0

            # extract content with w2v model
            with torch.no_grad():
                RVQ_1 = w2v_model.encode(wav_seg4c.to(device), n_q=1)
                RVQ_1_latent = w2v_model.quantizer.decode(RVQ_1, st=0)
            RVQ_1 = F.interpolate(RVQ_1.float(), int(RVQ_1.size(-1) * 1.6), mode='nearest')
            RVQ_1_latent = F.interpolate(RVQ_1_latent.float(), int(RVQ_1_latent.size(-1) * 1.6), mode='nearest')
            target_content_tokens = RVQ_1.squeeze()[:, single_side_context // 200:-single_side_context // 200].long()
            target_content_latents = RVQ_1_latent[..., single_side_context // 200:-single_side_context // 200]

            z = model.encoder(wav_seg)
            mel_prosody = gt_mel_seg[:, :20, :]
            z, quantized, commitment_loss, codebook_loss, timbre = model.quantizer(z, mel_prosody, gt_mel_seg, mels, mel_input_length)
            preds, rev_preds = model.fa_predictors(quantized, timbre)
            # z, codes, latents, commitment_loss, codebook_loss = model.quantizer(
            #     z, 12
            # )  # z is quantized latent

            pred_wave = model.decoder(z)

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

            optimizer.zero_grad()
            accelerator.backward(loss_d)
            grad_norm_d = torch.nn.utils.clip_grad_norm_(model.discriminator.parameters(), 10.0)
            optimizer.step('discriminator')
            optimizer.scheduler(key='discriminator')

            # generator loss
            signal = AudioSignal(wav_seg, sample_rate=16000)
            recons = AudioSignal(pred_wave, sample_rate=16000)
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

            # loss on predicting f0, uv, content and speaker
            # for f0 and energy prediction
            pred_f0, pred_uv = preds['f0'], preds['uv']
            c_pred_f0, c_pred_uv = rev_preds['content_f0'], rev_preds['content_uv']
            r_pred_f0, r_pred_uv = rev_preds['res_f0'], rev_preds['res_uv']

            common_min_size = min(pred_f0.size(-2), f0_targets.size(-1))
            f0_targets = f0_targets[..., :common_min_size]
            real_norm = real_norm[..., :common_min_size]


            f0_loss = F.smooth_l1_loss(f0_targets, pred_f0.squeeze(-1)[..., :common_min_size])
            uv_loss = F.smooth_l1_loss(real_norm, pred_uv.squeeze(-1)[..., :common_min_size])
            c_f0_loss = F.smooth_l1_loss(f0_targets, c_pred_f0.squeeze(-1)[..., :common_min_size]) if c_pred_f0 is not None else torch.FloatTensor([0]).to(device)
            c_uv_loss = torch.FloatTensor([0]).to(device) # content should contain energy information so do not reverse this one
            r_f0_loss = F.smooth_l1_loss(f0_targets, r_pred_f0.squeeze(-1)[..., :common_min_size]) if r_pred_f0 is not None else torch.FloatTensor([0]).to(device)
            r_uv_loss = F.smooth_l1_loss(real_norm, r_pred_uv.squeeze(-1)[..., :common_min_size]) if r_pred_uv is not None else torch.FloatTensor([0]).to(device)

            tot_f0_loss = f0_loss + c_f0_loss + r_f0_loss
            tot_uv_loss = uv_loss + c_uv_loss + r_uv_loss

            # loss on predicting content
            pred_content = preds['content']
            p_pred_content = rev_preds['prosody_content']
            r_pred_content = rev_preds['res_content']

            # target_content_tokens = target_content_tokens[..., :common_min_size]
            # content_loss = F.cross_entropy(pred_content.transpose(1, 2), target_content_tokens, ignore_index=-1)
            # p_content_loss = F.cross_entropy(p_pred_content.transpose(1, 2), target_content_tokens, ignore_index=-1) if p_pred_content is not None else torch.FloatTensor([0]).to(device)
            # r_content_loss = F.cross_entropy(r_pred_content.transpose(1, 2), target_content_tokens, ignore_index=-1) if r_pred_content is not None else torch.FloatTensor([0]).to(device)

            target_content_latents = target_content_latents[..., :common_min_size]

            content_loss = F.l1_loss(pred_content.transpose(1, 2)[..., :common_min_size], target_content_latents)
            p_content_loss = F.l1_loss(p_pred_content.transpose(1, 2)[..., :common_min_size], target_content_latents) if p_pred_content is not None else torch.FloatTensor([0]).to(device)
            r_content_loss = F.l1_loss(r_pred_content.transpose(1, 2)[..., :common_min_size], target_content_latents) if r_pred_content is not None else torch.FloatTensor([0]).to(device)

            tot_content_loss = content_loss + p_content_loss + r_content_loss

            # top 10 accuracy
            # content_top10_acc = content_metric(pred_content.transpose(1, 2), target_content_tokens)
            # p_content_top10_acc = content_metric(p_pred_content.transpose(1, 2), target_content_tokens) if p_pred_content is not None else torch.FloatTensor([0]).to(device)
            # r_content_top10_acc = content_metric(r_pred_content.transpose(1, 2), target_content_tokens) if r_pred_content is not None else torch.FloatTensor([0]).to(device)

            # loss on predicting speaker
            spk_pred_logits = preds['timbre']
            x_spk_pred_logits = rev_preds['x_timbre']
            spk_loss = spk_criterion(spk_pred_logits, speaker_labels)
            x_spk_loss = spk_criterion(x_spk_pred_logits, speaker_labels)

            tot_spk_loss = spk_loss + x_spk_loss

            loss_gen_all = mel_loss * 15.0 + loss_feature * 1.0 + loss_g * 1.0 + commitment_loss * 0.25 + codebook_loss * 1.0 \
                            + tot_f0_loss * 1.0 + tot_uv_loss * 1.0 + tot_content_loss * 5.0 + tot_spk_loss * 1.0

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
                    print("f0 Loss: %.4f, uv Loss: %.4f, content Loss: %.4f, spk Loss: %.4f" % (
                        f0_loss.item(), uv_loss.item(), content_loss.item(), spk_loss.item()))
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
                    # writer.add_scalar('pred/content_top10_acc', content_top10_acc.item(), iters)

                    writer.add_scalar('rev_pred/c_f0_loss', c_f0_loss.item(), iters)
                    writer.add_scalar('rev_pred/c_uv_loss', c_uv_loss.item(), iters)
                    writer.add_scalar('rev_pred/p_content_loss', p_content_loss.item(), iters)
                    writer.add_scalar('rev_pred/r_content_loss', r_content_loss.item(), iters)
                    writer.add_scalar('rev_pred/r_f0_loss', r_f0_loss.item(), iters)
                    writer.add_scalar('rev_pred/r_uv_loss', r_uv_loss.item(), iters)
                    writer.add_scalar('rev_pred/x_spk_loss', x_spk_loss.item(), iters)
                    # writer.add_scalar('rev_pred/p_content_top10_acc', p_content_top10_acc.item(), iters)
                    # writer.add_scalar('rev_pred/r_content_top10_acc', r_content_top10_acc.item(), iters)


                    print('Time elasped:', time.time() - start_time)
            if iters % (log_interval * 10) == 0 and accelerator.is_main_process:
                with torch.no_grad():
                    writer.add_audio('train/gt_audio', wav_seg[0], iters, sample_rate=16000)
                    writer.add_audio('train/pred_audio', pred_wave[0], iters, sample_rate=16000)

            if iters % (log_interval * 1000) == 0 and accelerator.is_main_process:
                with torch.no_grad():
                    # put ground truth audio
                    writer.add_audio('full/gt_audio', waves[0], iters, sample_rate=16000)

                    # without timbre norm
                    z = model.encoder(torch.from_numpy(waves[0])[None, None, ...].to(device).float())
                    mel_prosody = mels[0:1, :20, :z.size(-1)]
                    z, quantized, commitment_loss, codebook_loss, timbre = model.quantizer(z, mel_prosody,
                                                                                           mels[0:1, :, :z.size(-1)],
                                                                                           mels[0:1, :, :z.size(-1)],
                                                                                           mel_input_length[0:1])

                    z2 = model.encoder(torch.from_numpy(waves[1])[None, None, ...].to(device).float())
                    mel_prosody2 = mels[1:2, :20, :z2.size(-1)]
                    z2, quantized2, commitment_loss2, codebook_loss2, timbre2 = model.quantizer(z2, mel_prosody2,
                                                                                            mels[1:2, :, :z2.size(-1)],
                                                                                            mels[1:2, :, :z2.size(-1)],
                                                                                            mel_input_length[1:2])
                    # zero_timbre = torch.randn_like(timbre)
                    # zero_style = model.quantizer.timbre_linear(zero_timbre).unsqueeze(2)  # (B, 2d, 1)
                    # zero_gamma, zero_beta = zero_style.chunk(2, 1)  # (B, d, 1)

                    # p_pred_wave = model.decoder(model.quantizer.timbre_norm(quantized[0].transpose(1, 2)).transpose(1, 2) * zero_gamma + zero_beta)
                    # c_pred_wave = model.decoder(model.quantizer.timbre_norm(quantized[1].transpose(1, 2)).transpose(1, 2) * zero_gamma + zero_beta)
                    # r_pred_wave = model.decoder(model.quantizer.timbre_norm(quantized[2].transpose(1, 2)).transpose(1, 2) * zero_gamma + zero_beta)
                    # pc_pred_wave = model.decoder(model.quantizer.timbre_norm((quantized[0] + quantized[1]).transpose(1, 2)).transpose(1, 2) * zero_gamma + zero_beta)
                    # pr_pred_wave = model.decoder(model.quantizer.timbre_norm((quantized[0] + quantized[2]).transpose(1, 2)).transpose(1, 2) * zero_gamma + zero_beta)
                    # pcr_pred_wave = model.decoder(model.quantizer.timbre_norm((quantized[0] + quantized[1] + quantized[2]).transpose(1, 2)).transpose(1, 2) * zero_gamma + zero_beta)
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

                    writer.add_audio('partial/pred_audio_p', p_pred_wave[0], iters, sample_rate=16000)
                    writer.add_audio('partial/pred_audio_c', c_pred_wave[0], iters, sample_rate=16000)
                    writer.add_audio('partial/pred_audio_r', r_pred_wave[0], iters, sample_rate=16000)
                    writer.add_audio('partial/pred_audio_pc', pc_pred_wave[0], iters, sample_rate=16000)
                    writer.add_audio('partial/pred_audio_pr', pr_pred_wave[0], iters, sample_rate=16000)
                    writer.add_audio('partial/pred_audio_pcr', pcr_pred_wave[0], iters, sample_rate=16000)
                    writer.add_audio('full/pred_audio', full_pred_wave[0], iters, sample_rate=16000)

                    writer.add_audio('vc/ref_audio', waves[1], iters, sample_rate=16000)
                    writer.add_audio('vc/pred_audio', vc_pred_wave[0], iters, sample_rate=16000)

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
            iters = iters + 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='configs/config.yml')
    args = parser.parse_args()
    main(args)