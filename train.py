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
from optimizers import build_optimizer
import time

from accelerate import Accelerator
from accelerate.utils import LoggerType
from accelerate import DistributedDataParallelKwargs

from torch.utils.tensorboard import SummaryWriter

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

    train_dataloader = build_dataloader(train_path,
                                        root_path,
                                        min_length=min_length,
                                        batch_size=batch_size,
                                        batch_length=None,
                                        num_workers=0,
                                        rank=accelerator.local_process_index,
                                        world_size=accelerator.num_processes,
                                        dataset_config={},
                                        device=device,
                                        n_repeats=1,
                                        return_dict=True)


    with accelerator.main_process_first():
        pitch_extractor = load_F0_models(config['F0_path'])

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


    for epoch in range(start_epoch, epochs):
        start_time = time.time()
        train_dataloader.sampler.set_epoch(epoch)
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
            mel_prosody = mels[:, :20, :]
            vq_post_emb, vq_id, commit_loss, quantized, spk_embs = model.fa_decoder(enc_out, mel_prosody, eval_vq=False, vq=True)

            # random slice segments
            segment_len = min(max_frame_len, enc_out.size(2))


            out = model.fa_decoder(enc_out, mel_prosody, eval_vq=False, vq=False, quantized=quantized, speaker_embedding=spk_embs)

            optimizer.zero_grad()
            accelerator.backward(loss)
            grad_norm_g = torch.nn.utils.clip_grad_norm_(model.pllm.parameters(), 1000.)
            grad_norm_g2 = torch.nn.utils.clip_grad_norm_(model.prosody_encoder.parameters(), 1000.)
            # grad_norm_g3 = torch.nn.utils.clip_grad_norm_(model.prosody_resampler.parameters(), 1000.)
            grad_norm_g4 = torch.nn.utils.clip_grad_norm_(model.p_bert_encoder.parameters(), 1000.)
            # torch.nn.utils.clip_grad_value_(model.pllm.parameters(), 1000.)
            # torch.nn.utils.clip_grad_value_(model.prosody_encoder.parameters(), 1000.)
            # torch.nn.utils.clip_grad_value_(model.prosody_resampler.parameters(), 1000.)
            # torch.nn.utils.clip_grad_value_(model.p_bert_encoder.parameters(), 1000.)
            optimizer.step('pllm')
            optimizer.step('prosody_encoder')
            # optimizer.step('prosody_resampler')
            optimizer.step('p_bert_encoder')

            # optimizer.step()
            # train time count end
            train_time_per_step = time.time() - train_start_time

            if iters % log_interval == 0 and accelerator.is_main_process:
                with torch.no_grad():
                    cur_lr = optimizer.schedulers['pllm'].get_last_lr()[0] if i != 0 else 0
                    # log print and tensorboard
                    log_print("Epoch %d, Iteration %d, Loss: %.4f, pitch_Top10_Acc: %.4f, duration_ce_loss: %.4f, lr: %.6f, Time: %.2f" %
                              (epoch, iters, loss.item(), pitch_acc.item(), duration_ce_loss.item(), cur_lr, train_time_per_step), logger)
                    writer.add_scalar('train/loss', loss, iters)
                    writer.add_scalar('train/pitch_loss', pitch_loss, iters)
                    writer.add_scalar('train/duration_loss', duration_loss, iters)
                    writer.add_scalar('train/pitch_acc', pitch_acc, iters)
                    writer.add_scalar('train/duration_ce_loss', duration_ce_loss, iters)
                    writer.add_scalar('train/lr', cur_lr, iters)
                    writer.add_scalar('train/time', train_time_per_step, iters)

                    writer.add_scalar('grad_norm/text_encoder', grad_norm_g, iters)
                    writer.add_scalar('grad_norm/prosody_encoder', grad_norm_g2, iters)
                    # writer.add_scalar('grad_norm/prosody_resampler', grad_norm_g3, iters)
                    writer.add_scalar('grad_norm/p_bert_encoder', grad_norm_g4, iters)


                    print('Time elasped:', time.time() - start_time)
            if iters % (log_interval * 10) == 0 and accelerator.is_main_process:

                with torch.no_grad():
                    # plot attention map
                    attn_image = get_image(gt_attn.transpose(0, 1).cpu().numpy())
                    writer.add_figure('train/attn', attn_image, iters)
                    # plot predicted duration
                    round_pred_dur = pred_dur.round().long()
                    pred_aln_trg = torch.zeros(len(round_pred_dur), int(round_pred_dur.sum().data)).to(device)
                    c_frame = 0
                    for i in range(pred_aln_trg.size(0)):
                        pred_aln_trg[i, c_frame:c_frame + int(round_pred_dur[i].data)] = 1
                        c_frame += int(round_pred_dur[i].data)
                    pred_attn_image = get_image(pred_aln_trg.cpu().numpy().squeeze())
                    writer.add_figure('train/pred_attn', pred_attn_image, iters)

                    # plot predicted pitch curve
                    # plot gt f0 curve to a image
                    gt_f0s = gt_f0_codes
                    pred_f0s = pred_pitch_codes
                    fig1, ax1 = plt.subplots()
                    ax1 = plt.plot(gt_f0s.cpu().numpy(), label='GT F0')
                    writer.add_figure('train/f0', fig1, iters)
                    fig2, ax2 = plt.subplots()
                    ax2 = plt.plot(pred_f0s.cpu().numpy(), label='Pred F0')
                    writer.add_figure('train/pred_f0', fig2, iters)

            if iters % save_interval == 0 and accelerator.is_main_process:
                print('Saving..')
                state = {
                    'net': {key: model[key].state_dict() for key in model},
                    'optimizer': optimizer.state_dict(),
                    'scheduler': optimizer.scheduler_state_dict(),
                    'iters': iters,
                    'epoch': epoch,
                }
                save_path = osp.join(log_dir, 'PLLMv2_epoch_%05d_step_%05d.pth' % (epoch, iters))
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