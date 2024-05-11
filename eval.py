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

    log_dir = config['log_dir'] + '/eval'
    if not osp.exists(log_dir): os.makedirs(log_dir, exist_ok=True)
    shutil.copy(config_path, osp.join(log_dir, osp.basename(config_path)))
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True, broadcast_buffers=False)
    accelerator = Accelerator(project_dir=log_dir, split_batches=True, kwargs_handlers=[ddp_kwargs])
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

    val_dataloader = build_dataloader(val_path,
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

    model_params = recursive_munch(config['model_params'])
    model = build_model(model_params)

    _ = [model[key].to(device) for key in model]

    model, _, start_epoch, iters = load_checkpoint(model, None, config['pretrained_model'],
          load_only_params=True, ignore_modules=[], is_distributed=accelerator.num_processes > 1)

    _ = [model[key].eval() for key in model]

    for i, batch in enumerate(val_dataloader):

        waves = batch[0]
        paths = batch[-1]
        batch = [b.to(device, non_blocking=True) if type(b) == torch.Tensor else b for b in batch[1:-1]]
        texts, input_lengths, mels, mel_input_length, speaker_labels, langs = batch
        # note that the mel spec here must be sr=24000, n_fft=2048, hop_lenght=300, win_length=1200 to be compatible with f0 extractor
        bsz = len(waves)
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

            p_pred_wave = model.decoder(quantized[0])
            c_pred_wave = model.decoder(quantized[1])
            r_pred_wave = model.decoder(quantized[2])
            pc_pred_wave = model.decoder(quantized[0] + quantized[1])
            pr_pred_wave = model.decoder(quantized[0] + quantized[2])
            pcr_pred_wave = model.decoder(quantized[0] + quantized[1] + quantized[2])
            full_pred_wave = model.decoder(z)

            writer.add_audio('partial/pred_audio_p', p_pred_wave[0], iters, sample_rate=16000)
            writer.add_audio('partial/pred_audio_c', c_pred_wave[0], iters, sample_rate=16000)
            writer.add_audio('partial/pred_audio_r', r_pred_wave[0], iters, sample_rate=16000)
            writer.add_audio('partial/pred_audio_pc', pc_pred_wave[0], iters, sample_rate=16000)
            writer.add_audio('partial/pred_audio_pr', pr_pred_wave[0], iters, sample_rate=16000)
            writer.add_audio('partial/pred_audio_pcr', pcr_pred_wave[0], iters, sample_rate=16000)
            writer.add_audio('full/pred_audio', full_pred_wave[0], iters, sample_rate=16000)

            for bib in range(1, min(5, bsz)):
                x = quantized[0] + quantized[1] + quantized[2]

                z2 = model.encoder(torch.from_numpy(waves[bib])[None, None, ...].to(device).float())
                mel_prosody2 = mels[1:2, :20, :z2.size(-1)]
                z2, quantized2, commitment_loss2, codebook_loss2, timbre2 = model.quantizer(z2, mel_prosody2,
                                                                                            mels[bib:bib+1, :, :z2.size(-1)],
                                                                                            mels[bib:bib+1, :, :z2.size(-1)],
                                                                                            mel_input_length[bib:bib+1])

                style2 = model.quantizer.timbre_linear(timbre2).unsqueeze(2)  # (B, 2d, 1)
                gamma, beta = style2.chunk(2, 1)  # (B, d, 1)
                x = x.transpose(1, 2)
                x = model.quantizer.timbre_norm(x)
                x = x.transpose(1, 2)
                x = x * gamma + beta
                vc_pred_wave = model.decoder(x)
                writer.add_audio(f'vc_ref/audio_{bib}', waves[bib], iters, sample_rate=16000)
                writer.add_audio(f'vc_pred/audio_{bib}', vc_pred_wave[0], iters, sample_rate=16000)

        iters = iters + 1
        if iters > 10:
            exit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='configs/config.yml')
    args = parser.parse_args()
    main(args)