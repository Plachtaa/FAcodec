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


config = yaml.safe_load(open('configs/config_redecoder.yml'))
model_params = recursive_munch(config['model_params'])
model = build_model(model_params, stage="redecoder")
codec_encoder = build_model(model_params, stage="encoder")

model, _, _, _ = load_checkpoint(model, None, "./FAredecoder_epoch_00010_step_1426000_wavenet_noncausal_c1.pth",
                  load_only_params=True, ignore_modules=[], is_distributed=False)

codec_encoder, _, _, _ = load_checkpoint(codec_encoder, None, "./temp_ckpt.pth",
                  load_only_params=True, ignore_modules=[], is_distributed=False)

_ = [model[key].eval() for key in model]
_ = [model[key].to(device) for key in model]
_ = [codec_encoder[key].eval() for key in codec_encoder]
_ = [codec_encoder[key].to(device) for key in codec_encoder]

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

@torch.no_grad()
def main():
    source = "./test_waves/dingzhen_333.wav"
    target = "./test_waves/teio_0.wav"
    source_audio = librosa.load(source, sr=24000)[0]
    ref_audio = librosa.load(target, sr=24000)[0]
    # crop only the first 30 seconds
    source_audio = source_audio[:24000 * 30]
    source_audio = torch.tensor(source_audio).unsqueeze(0).float().to(device)

    ref_audio = ref_audio[:24000 * 30]
    ref_audio = torch.tensor(ref_audio).unsqueeze(0).float().to(device)

    # without timbre norm
    z = codec_encoder.encoder(source_audio[None, ...].to(device).float())
    z, quantized, commitment_loss, codebook_loss, timbre, codes = codec_encoder.quantizer(z,
                                                                           source_audio[None, ...].to(device).float(),
                                                                           torch.ones(1).to(device).bool(),
                                                                           torch.ones(1).to(device).bool(),
                                                                           n_c=2, return_codes=True)

    z2 = codec_encoder.encoder(ref_audio[None, ...].to(device).float())
    z2, quantized2, commitment_loss2, codebook_loss2, timbre2, codes2 = codec_encoder.quantizer(z2,
                                                                                ref_audio[None, ...].to(device).float(),
                                                                                torch.ones(1).to(device).bool(),
                                                                                torch.zeros(1).to(device).bool(),
                                                                                n_c=2, return_codes=True)
    z = model.encoder(codes[0], codes[1], timbre, use_p_code=False, n_c=1)
    full_pred_wave = model.decoder(z)
    z2 = model.encoder(codes[0], codes[1], timbre2, use_p_code=False, n_c=1)
    full_pred_wave2 = model.decoder(z2)

    os.makedirs("reconstructed", exist_ok=True)
    source_name = source.split("/")[-1].split(".")[0]
    target_name = target.split("/")[-1].split(".")[0]
    torchaudio.save(f"reconstructed/redecoder_full_pred_wave_{source_name}.wav", full_pred_wave[0].cpu(), 24000)
    torchaudio.save(f"reconstructed/redecoder_vc_pred_wave_{source_name}_{target_name}.wav", full_pred_wave2[0].cpu(), 24000)


if __name__ == "__main__":
    main()