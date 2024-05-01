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

from meldataset import build_dataloader, mel_spectrogram
from modules.commons import *
from losses import *
from optimizers import build_optimizer
import time

from accelerate import Accelerator
from accelerate.utils import LoggerType
from accelerate import DistributedDataParallelKwargs

from torch.utils.tensorboard import SummaryWriter
import torchaudio

import logging
from accelerate.logging import get_logger


logger = get_logger(__name__, log_level="DEBUG")
# torch.autograd.set_detect_anomaly(True)

def main(args):
    config_path = args.config_path
    config = yaml.safe_load(open(config_path))

    log_dir = config['log_dir'] + '/v1'
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

    epochs = config.get('epochs_1st', 200)
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

    scheduler_params = {
        "warmup_steps": 200,
        "base_lr": 0.0002,
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

    blank_index = train_dataloader.dataset.text_cleaner.word_index_dictionary[" "]  # get blank index
    phn_criterion = {
        "ce": nn.CrossEntropyLoss(ignore_index=-1),
        "ctc": torch.nn.CTCLoss(blank=blank_index, zero_infinity=True),
    }
    spk_criterion = FocalLoss(gamma=2).to(device)

    train_dataloader = accelerator.prepare(train_dataloader)


    for epoch in range(start_epoch, epochs):
        start_time = time.time()
        # train_dataloader.sampler.set_epoch(epoch)
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

            max_wave_length = max([len(wave) for wave in waves])
            wave_tensors = [torch.FloatTensor(wave).to(device) for wave in waves]
            wave_tensors = torch.stack([F.pad(wave, (0, max_wave_length - len(wave))) for wave in wave_tensors], dim=0).to(device)
            wave_tensors = wave_tensors.unsqueeze(1)
            enc_out = model.fa_encoder(wave_tensors)
            mels = mels[..., :enc_out.size(2)]
            mel_input_length = torch.clamp(mel_input_length, 0, enc_out.size(2))
            mel_prosody = mels[:, :20, :]
            vq_post_emb, vq_id, commit_loss, quantized, spk_embs = model.fa_decoder(enc_out, mel_prosody, eval_vq=False, vq=True)

            # get clips
            mel_seg_len = min([int(mel_input_length.min().item()), max_frame_len])

            en = [[], [], []]
            gt_mel_seg = []
            wav_seg = []
            # f0 = []

            for bib in range(len(mel_input_length)):
                mel_length = int(mel_input_length[bib].item())

                random_start = np.random.randint(0, mel_length - mel_seg_len)
                for q, layer_q in enumerate(quantized):
                    en[q].append(layer_q[bib, :, random_start:random_start + mel_seg_len])
                gt_mel_seg.append(mels[bib, :, random_start:random_start + mel_seg_len])

                y = waves[bib][random_start * 200:(random_start + mel_seg_len) * 200]
                wav_seg.append(torch.from_numpy(y).to(device))
                # # F0
                # f0.append(F0s[bib, (random_start * 2):((random_start + mel_len) * 2)])

            en = [torch.stack(e) for e in en]
            gt_mel_seg = torch.stack(gt_mel_seg).detach()
            # f0 = torch.stack(f0).detach()

            wav_seg = torch.stack(wav_seg).float().detach()

            with torch.no_grad():
                real_norm = log_norm(mels.unsqueeze(1)).squeeze(1).detach()
                F0_real, _, _ = pitch_extractor(mels.unsqueeze(1))
                # F0_real = F0_real.unsqueeze(0) if batch_size == 1 else F0_real


            out = model.fa_decoder(None, None, eval_vq=False, vq=False, quantized=en, speaker_embedding=spk_embs)
            pred_wave = out['audio']

            # generator loss
            wav_seg = wav_seg.unsqueeze(1)
            if iters >= discriminator_iter_start:
                optimizer.zero_grad("mrd")
                optimizer.zero_grad("mpd")
                # MPD
                y_df_hat_r, y_df_hat_g, _, _ = model.mpd(wav_seg, pred_wave.detach())
                loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(y_df_hat_r, y_df_hat_g)

                # MSD
                y_ds_hat_r, y_ds_hat_g, _, _ = model.mrd(wav_seg, pred_wave.detach())
                loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

                loss_disc_all = loss_disc_s + loss_disc_f
                accelerator.backward(loss_disc_all)
                optimizer.step("mrd")
                optimizer.step("mpd")

                optimizer.zero_grad("mrd")
                optimizer.zero_grad("mpd")
            else:
                loss_disc_all = torch.FloatTensor([0]).to(device)
                grad_norm_mpd = 0.
                grad_norm_mrd = 0.


            y_g_hat_mel = mel_spectrogram(pred_wave.squeeze(1), n_fft=1024, num_mels=80, sampling_rate=16000,
                                        hop_size=200,
                                        win_size=800,
                                        fmin=0,
                                        fmax=8000,
                                     )
            gt_mel = mel_spectrogram(wav_seg.squeeze(1), n_fft=1024, num_mels=80, sampling_rate=16000,
                                        hop_size=200,
                                        win_size=800,
                                        fmin=0,
                                        fmax=8000,
                                     )
            mel_loss = F.l1_loss(y_g_hat_mel, gt_mel)
            # for f0 and energy prediction
            prosody_latent = quantized[0]
            content_latent = quantized[1]
            res_latent = quantized[2]
            pred_f0, pred_uv = model.prosody_f0n_predictor(prosody_latent)
            c_pred_f0, c_pred_uv = model.content_f0n_predictor(content_latent)
            r_pred_f0, r_pred_uv = model.res_f0n_predictor(res_latent)
            f0_loss, uv_loss, c_f0_loss, c_uv_loss, r_f0_loss, r_uv_loss = 0, 0, 0, 0, 0, 0
            for bib, mel_len in enumerate(mel_input_length):
                f0_loss += F.smooth_l1_loss(F0_real[bib, :mel_len], pred_f0[bib, :mel_len].squeeze(-1))
                uv_loss += F.smooth_l1_loss(real_norm[bib, :mel_len], pred_uv[bib, :mel_len].squeeze(-1))
                c_f0_loss += F.smooth_l1_loss(F0_real[bib, :mel_len], c_pred_f0[bib, :mel_len].squeeze(-1))
                c_uv_loss += F.smooth_l1_loss(real_norm[bib, :mel_len], c_pred_uv[bib, :mel_len].squeeze(-1))
                r_f0_loss += F.smooth_l1_loss(F0_real[bib, :mel_len], r_pred_f0[bib, :mel_len].squeeze(-1))
                r_uv_loss += F.smooth_l1_loss(real_norm[bib, :mel_len], r_pred_uv[bib, :mel_len].squeeze(-1))
            f0_loss, uv_loss, c_f0_loss, c_uv_loss, r_f0_loss, r_uv_loss = (
                f0_loss / len(mel_input_length), uv_loss / len(mel_input_length), c_f0_loss / len(mel_input_length),
                c_uv_loss / len(mel_input_length), r_f0_loss / len(mel_input_length), r_uv_loss / len(mel_input_length))

            tot_f0_loss, tot_uv_loss = (f0_loss + c_f0_loss + r_f0_loss)/3, (uv_loss + c_uv_loss + r_uv_loss)/3

            # for phoneme prediction

            ppgs, s2s_pred, s2s_attn = model.content_phoneme_predictor(content_latent, texts, langs,
                                                                       mel_lens=mel_input_length, text_lens=input_lengths)
            ctc_loss = phn_criterion['ctc'](ppgs.log_softmax(dim=2).transpose(0, 1), texts, mel_input_length, input_lengths)
            s2s_loss = 0
            for _s2s_pred, _text_input, _text_length in zip(s2s_pred, texts, input_lengths):
                s2s_loss += phn_criterion['ce'](_s2s_pred[:_text_length], _text_input[:_text_length])
            s2s_loss /= texts.size(0)

            if accelerator.num_processes == 1:
                p_ppgs, p_s2s_pred, p_s2s_attn = model.prosody_phoneme_predictor[1](model.prosody_phoneme_predictor[0](prosody_latent),
                                                        texts, langs, mel_lens=mel_input_length, text_lens=input_lengths)
            else:
                p_ppgs, p_s2s_pred, p_s2s_attn = model.prosody_phoneme_predictor.module[1](model.prosody_phoneme_predictor.module[0](prosody_latent),
                                                        texts, langs, mel_lens=mel_input_length, text_lens=input_lengths)
            p_ctc_loss = phn_criterion['ctc'](p_ppgs.log_softmax(dim=2).transpose(0, 1), texts, mel_input_length, input_lengths)
            p_s2s_loss = 0
            for _s2s_pred, _text_input, _text_length in zip(p_s2s_pred, texts, input_lengths):
                p_s2s_loss += phn_criterion['ce'](_s2s_pred[:_text_length], _text_input[:_text_length])
            p_s2s_loss /= texts.size(0)

            if accelerator.num_processes == 1:
                r_ppgs, r_s2s_pred, r_s2s_attn = model.res_phoneme_predictor[1](model.res_phoneme_predictor[0](res_latent),
                                                        texts, langs, mel_lens=mel_input_length, text_lens=input_lengths)
            else:
                r_ppgs, r_s2s_pred, r_s2s_attn = model.res_phoneme_predictor.module[1](model.res_phoneme_predictor.module[0](res_latent),
                                                        texts, langs, mel_lens=mel_input_length, text_lens=input_lengths)
            r_ctc_loss = phn_criterion['ctc'](r_ppgs.log_softmax(dim=2).transpose(0, 1), texts, mel_input_length, input_lengths)
            r_s2s_loss = 0
            for _s2s_pred, _text_input, _text_length in zip(r_s2s_pred, texts, input_lengths):
                r_s2s_loss += phn_criterion['ce'](_s2s_pred[:_text_length], _text_input[:_text_length])
            r_s2s_loss /= texts.size(0)

            tot_ctc_loss = (ctc_loss + p_ctc_loss + r_ctc_loss)/3
            tot_s2s_loss = (s2s_loss + p_s2s_loss + r_s2s_loss)/3

            # speaker prediction loss
            spk_pred_logits = model.timbre_predictor(spk_embs, speaker_labels)
            bsz = quantized[2].shape[0]
            res_mask = np.random.choice(
                [0, 1],
                size=bsz,
                p=[
                    0.75,
                    1 - 0.75,
                ],
            )
            res_mask = (
                torch.from_numpy(res_mask).unsqueeze(1).unsqueeze(1)
            )  # (B, 1, 1)
            res_mask = res_mask.to(
                device=quantized[2].device, dtype=quantized[2].dtype
            )
            x = (
                    quantized[0].detach()
                    + quantized[1].detach()
                    + quantized[2] * res_mask
            )
            x_spk_embs = model.x_timbre_encoder(x)[0]
            x_spk_pred_logits = model.x_timbre_predictor(x_spk_embs, speaker_labels)

            spk_loss = spk_criterion(spk_pred_logits, speaker_labels)
            x_spk_loss = spk_criterion(x_spk_pred_logits, speaker_labels)

            tot_spk_loss = (spk_loss + x_spk_loss)/2

            if iters >= discriminator_iter_start:
                # MPD loss
                y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = model.mpd(wav_seg, pred_wave)
                loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
                loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)

                # MRD loss
                y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = model.mrd(wav_seg, pred_wave)
                loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
                loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)
                loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + mel_loss * 5 \
                + tot_f0_loss * 1 + tot_uv_loss * 1 + tot_ctc_loss * 1 + tot_s2s_loss * 1 + tot_spk_loss * 1 + commit_loss.sum() * 0.15
            else:
                loss_gen_all = mel_loss * 5 \
                + tot_f0_loss * 1 + tot_uv_loss * 1 + tot_ctc_loss * 1 + tot_s2s_loss * 1 + tot_spk_loss * 1 + commit_loss.sum() * 0.15

            optimizer.zero_grad()
            accelerator.backward(loss_gen_all)
            grad_norm_g = torch.nn.utils.clip_grad_norm_(model.fa_encoder.parameters(), 1000.0)
            grad_norm_g2 = torch.nn.utils.clip_grad_norm_(model.fa_decoder.parameters(), 1000.0)
            grad_norm_g3 = torch.nn.utils.clip_grad_norm_(model.prosody_f0n_predictor.parameters(), 1000.0)
            grad_norm_g4 = torch.nn.utils.clip_grad_norm_(model.content_f0n_predictor.parameters(), 1000.0)
            grad_norm_g5 = torch.nn.utils.clip_grad_norm_(model.res_f0n_predictor.parameters(), 1000.0)
            grad_norm_g6 = torch.nn.utils.clip_grad_norm_(model.content_phoneme_predictor.parameters(), 1000.0)
            grad_norm_g7 = torch.nn.utils.clip_grad_norm_(model.prosody_phoneme_predictor.parameters(), 1000.0)
            grad_norm_g8 = torch.nn.utils.clip_grad_norm_(model.res_phoneme_predictor.parameters(), 1000.0)
            grad_norm_g9 = torch.nn.utils.clip_grad_norm_(model.timbre_predictor.parameters(), 1000.0)
            grad_norm_g10 = torch.nn.utils.clip_grad_norm_(model.x_timbre_encoder.parameters(), 1000.0)
            grad_norm_g11 = torch.nn.utils.clip_grad_norm_(model.x_timbre_predictor.parameters(), 1000.0)

            optimizer.step('fa_encoder')
            optimizer.step('fa_decoder')
            optimizer.step('prosody_f0n_predictor')
            optimizer.step('content_f0n_predictor')
            optimizer.step('res_f0n_predictor')
            optimizer.step('content_phoneme_predictor')
            optimizer.step('prosody_phoneme_predictor')
            optimizer.step('res_phoneme_predictor')
            optimizer.step('timbre_predictor')
            optimizer.step('x_timbre_encoder')
            optimizer.step('x_timbre_predictor')


            # optimizer.step()
            # train time count end
            train_time_per_step = time.time() - train_start_time

            if iters % log_interval == 0 and accelerator.is_main_process:
                with torch.no_grad():
                    cur_lr = optimizer.schedulers['fa_encoder'].get_last_lr()[0] if i != 0 else 0
                    # log print and tensorboard
                    print("Epoch %d, Iteration %d, Gen Loss: %.4f, Disc Loss: %.4f, Mel Loss: %.4f, F0 Loss: %.4f, UV Loss: %.4f, CTC Loss: %.4f, S2S Loss: %.4f, SPK Loss: %.4f, Time: %.4f" % (
                        epoch, iters, loss_gen_all.item(), loss_disc_all.item(), mel_loss.item(), f0_loss.item(), uv_loss.item(), ctc_loss.item(), s2s_loss.item(), spk_loss.item(), train_time_per_step))
                    writer.add_scalar('train/lr', cur_lr, iters)
                    writer.add_scalar('train/time', train_time_per_step, iters)

                    writer.add_scalar('grad_norm/fa_encoder', grad_norm_g, iters)
                    writer.add_scalar('grad_norm/fa_decoder', grad_norm_g2, iters)
                    writer.add_scalar('grad_norm/prosody_f0n_predictor', grad_norm_g3, iters)
                    writer.add_scalar('grad_norm/content_f0n_predictor', grad_norm_g4, iters)
                    writer.add_scalar('grad_norm/res_f0n_predictor', grad_norm_g5, iters)
                    writer.add_scalar('grad_norm/content_phoneme_predictor', grad_norm_g6, iters)
                    writer.add_scalar('grad_norm/prosody_phoneme_predictor', grad_norm_g7, iters)
                    writer.add_scalar('grad_norm/res_phoneme_predictor', grad_norm_g8, iters)
                    writer.add_scalar('grad_norm/timbre_predictor', grad_norm_g9, iters)
                    writer.add_scalar('grad_norm/x_timbre_encoder', grad_norm_g10, iters)
                    writer.add_scalar('grad_norm/x_timbre_predictor', grad_norm_g11, iters)

                    writer.add_scalar('train/loss_gen_all', loss_gen_all.item(), iters)
                    writer.add_scalar('train/loss_disc_all', loss_disc_all.item(), iters)
                    writer.add_scalar('train/mel_loss', mel_loss.item(), iters)
                    writer.add_scalar('train/f0_loss', f0_loss.item(), iters)
                    writer.add_scalar('train/uv_loss', uv_loss.item(), iters)
                    writer.add_scalar('train/ctc_loss', ctc_loss.item(), iters)
                    writer.add_scalar('train/s2s_loss', s2s_loss.item(), iters)
                    writer.add_scalar('train/spk_loss', spk_loss.item(), iters)

                    writer.add_scalar('rev/c_f0_loss', c_f0_loss.item(), iters)
                    writer.add_scalar('rev/c_uv_loss', c_uv_loss.item(), iters)
                    writer.add_scalar('rev/r_f0_loss', r_f0_loss.item(), iters)
                    writer.add_scalar('rev/r_uv_loss', r_uv_loss.item(), iters)
                    writer.add_scalar('rev/p_ctc_loss', p_ctc_loss.item(), iters)
                    writer.add_scalar('rev/p_s2s_loss', p_s2s_loss.item(), iters)
                    writer.add_scalar('rev/r_ctc_loss', r_ctc_loss.item(), iters)
                    writer.add_scalar('rev/r_s2s_loss', r_s2s_loss.item(), iters)
                    writer.add_scalar('rev/x_spk_loss', x_spk_loss.item(), iters)

                    writer.add_scalar('commit_loss/layer_0', commit_loss[0].item(), iters)
                    writer.add_scalar('commit_loss/layer_1', commit_loss[1].item(), iters)
                    writer.add_scalar('commit_loss/layer_2', commit_loss[2].item(), iters)
                    writer.add_scalar('commit_loss/layer_3', commit_loss[3].item(), iters)
                    writer.add_scalar('commit_loss/layer_4', commit_loss[4].item(), iters)
                    writer.add_scalar('commit_loss/layer_5', commit_loss[5].item(), iters)



                    print('Time elasped:', time.time() - start_time)
            if iters % (log_interval * 10) == 0 and accelerator.is_main_process:

                with torch.no_grad():
                    # put predicted audio
                    writer.add_audio('train/pred_audio', pred_wave[0], iters, sample_rate=16000)

                    # put ground truth audio
                    writer.add_audio('train/gt_audio', wav_seg[0], iters, sample_rate=16000)

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
            optimizer.scheduler(iters)
            accelerator.wait_for_everyone()
        accelerator.wait_for_everyone()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='configs/config.yml')
    args = parser.parse_args()
    main(args)