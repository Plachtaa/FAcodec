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
```

## Usage

## Model storage

## Extracting representations

## Zero-shot voice conversion

## Real-time voice conversion
