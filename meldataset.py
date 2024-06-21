# coding: utf-8
import os
import os.path as osp
import time
import random
import numpy as np
import random
import soundfile as sf
import librosa

import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader

import math

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
from torch.utils.data.distributed import DistributedSampler


np.random.seed(114514)
random.seed(114514)
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


def preprocess(wave):
    # wave = wave.unsqueeze(0)
    wave_tensor = torch.from_numpy(wave).float()
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    return mel_tensor


class PseudoDataset(torch.utils.data.Dataset):
    def __init__(self,
                 list_path,
                 sr=24000,
                 range=(1, 30), # length of the audio duration in seconds
                 ):

        self.data_list = [] # read your list path here
        self.sr = sr
        self.duration_range = range

    def __len__(self):
        # return len(self.data_list)
        return 100 # return a fixed number for testing

    def __getitem__(self, idx):
        # replace this with your own data loading
        # wave, sr = librosa.load(self.data_list[idx], sr=self.sr)
        wave = np.random.randn(self.sr * random.randint(*self.duration_range)).clamp(-1, 1)
        mel = preprocess(wave)
        return wave, mel


def collate(batch):
    # batch[0] = wave, mel, text, f0, speakerid
    batch_size = len(batch)

    # sort by mel length
    lengths = [b[1].shape[1] for b in batch]
    batch_indexes = np.argsort(lengths)[::-1]
    batch = [batch[bid] for bid in batch_indexes]

    nmels = batch[0][1].size(0)
    max_mel_length = max([b[1].shape[1] for b in batch])
    max_wave_length = max([b[0].size(0) for b in batch])

    mels = torch.zeros((batch_size, nmels, max_mel_length)).float() - 10
    waves = torch.zeros((batch_size, max_wave_length)).float()

    mel_lengths = torch.zeros(batch_size).long()
    wave_lengths = torch.zeros(batch_size).long()

    for bid, (wave, mel) in enumerate(batch):
        mel_size = mel.size(1)
        mels[bid, :, :mel_size] = mel
        waves[bid, : wave.size(0)] = wave
        mel_lengths[bid] = mel_size
        wave_lengths[bid] = wave.size(0)

    return waves, mels, wave_lengths, mel_lengths


def build_dataloader(
    rank=0,
    world_size=1,
    batch_size=32,
    num_workers=0,
    prefetch_factor=16,
):
    dataset = PseudoDataset() # replace this with your own dataset
    collate_fn = collate
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=114514,
    )
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        drop_last=True,
        collate_fn=collate_fn,
        pin_memory=True,
        prefetch_factor=prefetch_factor,
        # shuffle=True,
    )

    return data_loader

