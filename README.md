# FAcodec

This repository is a Pytorch implementation for the training of FAcodec, which was proposed in paper [NaturalSpeech 3: Zero-Shot Speech Synthesis
with Factorized Codec and Diffusion Models](https://arxiv.org/pdf/2403.03100)  

This implementation made some key modifications to the training pipeline, so that the requirements of any form of annotations, including 
transcripts, phoneme alignments, and speaker labels, are eliminated. All you need are simply raw speech files.  
With the new training pipeline, it is possible to train the model on more languages with more diverse timbre distributions.  
We release the code for training and inference, including a pretrained checkpoint on 50k hours speech data.

## Requirements
- Python 3.10

## Installation
```bash
pip install -r requirements.txt
```
If you want to train the model by yourself, install the following packages:
```bash
pip install nemo_toolkit['all']
pip install descript-audio-codec
```

## Model storage
We provide pretrained checkpoints on 50k hours speech data.  

| Model type        | Hugging Face Badge                                                                                                                   |
|-------------------|--------------------------------------------------------------------------------------------------------------------------------------|
| FAcodec           | [![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Model%20A-blue)](https://huggingface.co/Plachta/FAcodec)           |
| FAcodec redecoder | [![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Model%20B-blue)](https://huggingface.co/Plachta/FAcodec-redecoder) |
## Usage

### Encode & reconstruct
```bash
python reconstruct_redecoder.py --source <source_wav> --target <target_wav>
```
Model weights will be automatically downloaded from Hugging Face.  
For China mainland users, add additional environment variable to specify huggingface endpoint:
```bash
HF_ENDPOINT=https://huggingface.co python reconstruct_redecoder.py --source <source_wav> --target <target_wav>
```

## Extracting representations
WIP

## Zero-shot voice conversion
```bash
python reconstruct_redecoder.py --source <source_wav> --target <target_wav>
```
same as above, model weights will be automatically downloaded from Hugging Face.

## Real-time voice conversion
This codec is fully causal, so it can be used for real-time voice conversion.  
Script are still under development and will be released soon.
