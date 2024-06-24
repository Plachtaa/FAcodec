import shutil
import warnings
import argparse
import torch
import os
import yaml

warnings.simplefilter('ignore')

from modules.commons import *
from hf_utils import load_custom_model_from_hf
from losses import *
import time

import torchaudio
import librosa
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ckpt_path, config_path = load_custom_model_from_hf("Plachta/FAcodec")

config = yaml.safe_load(open(config_path))
model_params = recursive_munch(config['model_params'])
model = build_model(model_params)

ckpt_params = torch.load(ckpt_path, map_location="cpu")

for key in ckpt_params:
    model[key].load_state_dict(ckpt_params[key])

_ = [model[key].eval() for key in model]
_ = [model[key].to(device) for key in model]

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

@torch.no_grad()
def main(args):
    source = args.source
    source_audio = librosa.load(source, sr=24000)[0]
    # crop only the first 30 seconds
    source_audio = source_audio[:24000 * 30]
    source_audio = torch.tensor(source_audio).unsqueeze(0).float().to(device)

    # without timbre norm
    z = model.encoder(source_audio[None, ...].to(device).float())
    z, quantized, commitment_loss, codebook_loss, timbre = model.quantizer(z,
                                                                           source_audio[None, ...].to(device).float(),
                                                                           n_c=2)

    full_pred_wave = model.decoder(z)

    os.makedirs("reconstructed", exist_ok=True)
    source_name = source.split("/")[-1].split(".")[0]
    torchaudio.save(f"reconstructed/{source_name}.wav", full_pred_wave[0].cpu(), 24000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True)
    args = parser.parse_args()
    main(args)