import shutil
import warnings
import argparse
import torch
import os
import os.path as osp
import json
import yaml

warnings.simplefilter('ignore')

# load packages
import random

from modules.commons import *
import time

from torch.utils.tensorboard import SummaryWriter
import torchaudio
# from torchmetrics.classification import MulticlassAccuracy

import logging

import glob
from vc.models import MambaLMHeadModel
import librosa

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@torch.no_grad()
def main(args):
    codec_config_path = args.codec_config
    model_config_path = args.model_config
    codec_config = yaml.safe_load(open(codec_config_path, 'r'))
    config = json.load(open(model_config_path, 'r'))

    # load codec
    codec_model = build_model(recursive_munch(codec_config['model_params']), stage='codec')
    codec_model, _, _, _ = load_checkpoint(codec_model, None, config['codec_configs']['checkpoint_path'],
                                                           load_only_params=config.get('load_only_params',
                                                                                       True), ignore_modules=[],
                                                           is_distributed=False)
    _ = [codec_model[key].eval() for key in codec_model]
    _ = [codec_model[key].to(device) for key in codec_model]

    # load model
    model = build_model(config, stage='vc')
    model, _, _, _ = load_checkpoint(model, None, config['pretrained_model'],
                                     load_only_params=config.get('load_only_params', True), ignore_modules=[],
                                     is_distributed=False)
    _ = [model[key].eval() for key in model]
    _ = [model[key].to(device) for key in model]

    # generation_model = MambaLMHeadModel(model.vc_core, model.vc_head).to(device)

    source_audio = librosa.load("./test_waves/p232_005_noisy.wav", sr=24000)[0]
    # crop only the first 30 seconds
    source_audio = source_audio[:24000 * 30]
    waves = torch.from_numpy(source_audio[None, None, :]).to(device)

    z = codec_model.encoder(waves)
    codes, quantized = codec_model.quantizer.encode(z, waves)

    reference_audio = librosa.load("./test_waves/teio_1.wav", sr=24000)[0]
    reference_audio = reference_audio[:24000 * 30]
    reference_wave = torch.from_numpy(reference_audio[None, None, :]).to(device)
    reference_z = codec_model.encoder(reference_wave)
    reference_codes, reference_quantized = codec_model.quantizer.encode(reference_z, reference_wave)

    shifted_reference_codes = torch.cat([
        reference_codes[0],
        torch.cat([torch.zeros([1, reference_codes[0].size(1), 1]).to(device) + 1024, reference_codes[1][..., :-1]], dim=-1),
        torch.cat([torch.zeros([1, reference_codes[2].size(1), 1]).to(device) + 1024, reference_codes[2][..., :-1]], dim=-1)
    ], dim=1)

    shifted_codes = torch.cat([
        codes[0],
        torch.cat([torch.zeros([1, codes[0].size(1), 1]).to(device) + 1024, codes[1][..., :-1]], dim=-1),
        torch.cat([torch.zeros([1, codes[2].size(1), 1]).to(device) + 1024, codes[2][..., :-1]], dim=-1)
    ], dim=1)
    code_length = shifted_codes.size(-1)
    prompt_len = random.randint(int(code_length * 0.2), int(code_length * 0.5))
    shifted_code_prompt = shifted_codes[:, :, :prompt_len]
    shifted_code_target = shifted_codes[:, :, prompt_len:].clone()
    # shifted_code_target[:, 1:, :] = 1024
    out = model.vc_core.generate(
        vc_head = model.vc_head,
        prompt_ids=shifted_code_prompt.long(),
        input_ids=shifted_code_target.long(),
        top_k=1,
        top_p=-1.0,
        temperature=1.0,
    )

    vc_out = model.vc_core.generate(
        vc_head = model.vc_head,
        prompt_ids=shifted_reference_codes.long(),
        input_ids=shifted_codes.long(),
        top_k=1,
        top_p=-1.0,
        temperature=1.0,
    )

    # prediction_full = torch.cat([expanded_code_prompt[..., -1:], expanded_code_target], dim=-1) + \
    #                     out - 1024
    prediction_full = torch.cat([out[:, 0:1, :-1], out[:, 1:, 1:]], dim=1)
    prediction_full = prediction_full.long()

    pred_z, pred_quantized = codec_model.quantizer.decode(prediction_full)

    pred_wave = codec_model.decoder(pred_quantized[0] + pred_quantized[1] + pred_quantized[2])

    os.makedirs("vc_reconstructed", exist_ok=True)
    torchaudio.save("vc_reconstructed/pred_wave.wav", pred_wave[0].cpu(), 24000)

    gt_full = shifted_codes[:, :, prompt_len:]
    gt_full = gt_full.long()

    gt_z, gt_quantized = codec_model.quantizer.decode(gt_full)

    gt_wave = codec_model.decoder(gt_quantized[0] + gt_quantized[1] + gt_quantized[2])
    torchaudio.save("vc_reconstructed/gt_wave.wav", gt_wave[0].cpu(), 24000)

    vc_predition_full = torch.cat([vc_out[:, 0:1, :-1], vc_out[:, 1:, 1:]], dim=1)
    vc_predition_full = vc_predition_full.long()

    vc_pred_z, vc_pred_quantized = codec_model.quantizer.decode(vc_predition_full)

    vc_pred_wave = codec_model.decoder(vc_pred_quantized[0] + vc_pred_quantized[1] + vc_pred_quantized[2])
    torchaudio.save("vc_reconstructed/vc_pred_wave.wav", vc_pred_wave[0].cpu(), 24000)

    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Reconstruct the audio from the model')
    parser.add_argument('--codec-config', type=str, default='./configs/config_emilia.yml', help='Path to the codec config file')
    parser.add_argument('--model-config', type=str, default='./vc/configs/config.json', help='Path to the model config file')
    args = parser.parse_args()
    main(args)