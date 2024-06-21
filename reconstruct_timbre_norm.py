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

from modules.commons import *
from losses import *
from optimizers import build_optimizer
import time

import torchaudio
import librosa
from audiotools import AudioSignal
import glob


SPECT_PARAMS = {
    "n_fft": 2048,
    "win_length": 1200,
    "hop_length": 300,
}
MEL_PARAMS = {
    "n_mels": 80,
}

to_mel = torchaudio.transforms.MelSpectrogram(
    n_mels=MEL_PARAMS['n_mels'], **SPECT_PARAMS)
mean, std = -4, 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def preprocess(wave):
    # input is desired to be 16000hz, this operation resamples it to 24000hz
    # wave = wave.unsqueeze(0)
    wave_tensor = torch.from_numpy(wave).float()
    # wave_tensor = torchaudio.functional.resample(wave_tensor, 16000, 24000)
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    return mel_tensor


config = yaml.safe_load(open('configs/config_emilia_v3.yml'))
model_params = recursive_munch(config['model_params'])
model = build_model(model_params)

model, optimizer, start_epoch, iters = load_checkpoint(model, None, "./temp_ckpt.pth",
                  load_only_params=True, ignore_modules=[], is_distributed=False)

_ = [model[key].eval() for key in model]
_ = [model[key].to(device) for key in model]

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

@torch.no_grad()
def main():
    source = "./test_waves/suzuka_0.wav"
    target = "./test_waves/cafe_0.wav"
    source_audio = librosa.load(source, sr=24000)[0]
    ref_audio = librosa.load(target, sr=24000)[0]
    # crop only the first 30 seconds
    source_audio = source_audio[:24000 * 30]
    source_audio = torch.tensor(source_audio).unsqueeze(0).float().to(device)

    ref_audio = ref_audio[:24000 * 30]
    ref_audio = torch.tensor(ref_audio).unsqueeze(0).float().to(device)

    # without timbre norm
    z = model.encoder(source_audio[None, ...].to(device).float())
    z, quantized, commitment_loss, codebook_loss, timbre = model.quantizer(z,
                                                                           source_audio[None, ...].to(device).float(),
                                                                           torch.ones(1).to(device).bool(),
                                                                           torch.ones(1).to(device).bool(),
                                                                           n_c=2)

    z2 = model.encoder(ref_audio[None, ...].to(device).float())
    z2, quantized2, commitment_loss2, codebook_loss2, timbre2 = model.quantizer(z2,
                                                                                ref_audio[None, ...].to(device).float(),
                                                                                torch.ones(1).to(device).bool(),
                                                                                torch.zeros(1).to(device).bool(),
                                                                                n_c=2)

    full_pred_wave = model.decoder(z)
    x = quantized[1]# + quantized[1]# + quantized[2]
    style2 = model.quantizer.timbre_linear(timbre2).unsqueeze(2)  # (B, 2d, 1)
    gamma, beta = style2.chunk(2, 1)  # (B, d, 1)
    x = x.transpose(1, 2)
    x = model.quantizer.timbre_norm(x)
    x = x.transpose(1, 2)
    x = x * gamma + beta
    vc_pred_wave = model.decoder(x)

    os.makedirs("reconstructed", exist_ok=True)
    source_name = source.split("/")[-1].split(".")[0]
    target_name = target.split("/")[-1].split(".")[0]
    torchaudio.save(f"reconstructed/full_pred_wave_{source_name}.wav", full_pred_wave[0].cpu(), 24000)
    torchaudio.save(f"reconstructed/vc_pred_wave_{source_name}_{target_name}.wav", vc_pred_wave[0].cpu(), 24000)


if __name__ == "__main__":
    main()