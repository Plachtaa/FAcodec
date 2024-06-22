import gradio as gr
import torch
import torchaudio
import librosa
import numpy as np
import os
from huggingface_hub import hf_hub_download
import yaml
from modules.commons import recursive_munch, build_model

# setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# load model
def load_model(repo_id):
    ckpt_path = hf_hub_download(repo_id, "pytorch_model.bin", cache_dir="./checkpoints")
    config_path = hf_hub_download(repo_id, "config.yml", cache_dir="./checkpoints")

    config = yaml.safe_load(open(config_path))
    model_params = recursive_munch(config['model_params'])

    if "redecoder" in repo_id:
        model = build_model(model_params, stage="redecoder")
    else:
        model = build_model(model_params, stage="codec")

    ckpt_params = torch.load(ckpt_path, map_location="cpu")

    for key in model:
        model[key].load_state_dict(ckpt_params[key])
        model[key].eval()
        model[key].to(device)

    return model


# load models
codec_model = load_model("Plachta/FAcodec")
redecoder_model = load_model("Plachta/FAcodec-redecoder")


# preprocess audio
def preprocess_audio(audio_path, sr=24000):
    audio = librosa.load(audio_path, sr=sr)[0]
    # if audio has two channels, take the first one
    if len(audio.shape) > 1:
        audio = audio[0]
    audio = audio[:sr * 30]  # crop only the first 30 seconds
    return torch.tensor(audio).unsqueeze(0).float().to(device)


# audio reconstruction function
@torch.no_grad()
def reconstruct_audio(audio):
    source_audio = preprocess_audio(audio)

    z = codec_model.encoder(source_audio[None, ...])
    z, _, _, _, _ = codec_model.quantizer(z, source_audio[None, ...], n_c=2)

    reconstructed_wave = codec_model.decoder(z)

    return (24000, reconstructed_wave[0, 0].cpu().numpy())


# voice conversion function
@torch.no_grad()
def voice_conversion(source_audio, target_audio):
    source_audio = preprocess_audio(source_audio)
    target_audio = preprocess_audio(target_audio)

    z = codec_model.encoder(source_audio[None, ...])
    z, _, _, _, timbre, codes = codec_model.quantizer(z, source_audio[None, ...], n_c=2, return_codes=True)

    z_target = codec_model.encoder(target_audio[None, ...])
    _, _, _, _, timbre_target, _ = codec_model.quantizer(z_target, target_audio[None, ...], n_c=2, return_codes=True)

    z_converted = redecoder_model.encoder(codes[0], codes[1], timbre_target, use_p_code=False, n_c=1)
    converted_wave = redecoder_model.decoder(z_converted)

    return (24000, converted_wave[0, 0].cpu().numpy())


# gradio interface
def gradio_interface():
    with gr.Blocks() as demo:
        gr.Markdown(
            "# FAcodec reconstruction and voice conversion"
            "[![GitHub stars](https://img.shields.io/github/stars/username/repo-name.svg?style=social&label=Star&maxAge=2592000)](https://github.com/Plachtaa/FAcodec)"
            "FAcodec from [Natural Speech 3](https://arxiv.org/pdf/2403.03100). The checkpoint used in this demo is trained on an improved pipeline of "
            "where all kinds of annotations are not required, enabling the scale up of training data. <br>This model is "
            "trained on 50k hours of data with over 1 million speakers, largely improved timbre diversity compared to "
            "the [original FAcodec](https://huggingface.co/spaces/amphion/naturalspeech3_facodec)."
            "<br><br>This project is supported by [Amphion](https://github.com/open-mmlab/Amphion)"
        )

        with gr.Tab("reconstruction"):
            with gr.Row():
                input_audio = gr.Audio(type="filepath", label="Input audio")
                output_audio = gr.Audio(label="Reconstructed audio")
            reconstruct_btn = gr.Button("Reconstruct")
            reconstruct_btn.click(reconstruct_audio, inputs=[input_audio], outputs=[output_audio])

        with gr.Tab("voice conversion"):
            with gr.Row():
                source_audio = gr.Audio(type="filepath", label="Source audio")
                target_audio = gr.Audio(type="filepath", label="Reference audio")
                converted_audio = gr.Audio(label="Converted audio")
            convert_btn = gr.Button("Convert")
            convert_btn.click(voice_conversion, inputs=[source_audio, target_audio], outputs=[converted_audio])

    return demo


if __name__ == "__main__":
    iface = gradio_interface()
    iface.launch()