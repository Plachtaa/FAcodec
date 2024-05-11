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

import pandas as pd
from torch.utils.data.distributed import DistributedSampler
from librosa.filters import mel as librosa_mel_fn

_pad = "$"
_punctuation = ';:,.!?¡¿-…"«»“” '
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞→↗⇵↘'̩'ᵻ"
_expanded = ["\u0303", "\u032a", "\u2014", "1", "^"]
# Export all symbols:
symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa) + _expanded

lang_dict = {
    'en-us': 0,
}

dicts = {}
for i in range(len((symbols))):
    dicts[symbols[i]] = i


class TextCleaner:
    def __init__(self, dummy=None):
        self.word_index_dictionary = dicts

    def __call__(self, text):
        indexes = []
        text = text.replace("\n", "")
        for char in text:
            try:
                indexes.append(self.word_index_dictionary[char])
            except KeyError:
                print(text)
        return indexes


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
    # input is desired to be 16000hz, this operation resamples it to 24000hz
    # wave = wave.unsqueeze(0)
    wave_tensor = torch.from_numpy(wave).float()
    wave_tensor = torchaudio.functional.resample(wave_tensor, 16000, 24000)
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    return mel_tensor


class FilePathDataset(torch.utils.data.Dataset):
    def __init__(self,
                 list_path,
                 root_path,
                 sr=24000,
                 data_augmentation=False,
                 validation=False,
                 min_length=50,
                 max_ref_time=30,  # maximum reference audio length (in seconds)
                 max_ref_num=5,  # maximum number of reference samples
                 n_repeats=1,
                 return_dict=False,
                 ):

        data_list = open(list_path, 'r', encoding='utf-8').readlines()

        _data_list = [l[:-1].split('\t') if "\t" in l else l[:-1].split('|') for l in data_list]

        self.data_list = [d for d in _data_list if len(d) == 5]
        self.text_cleaner = TextCleaner()
        self.sr = sr

        self.df = pd.DataFrame(self.data_list)

        self.to_melspec = torchaudio.transforms.MelSpectrogram(**MEL_PARAMS)

        self.mean, self.std = -4, 4
        self.data_augmentation = data_augmentation and (not validation)

        self.min_length = min_length

        self.root_path = root_path

        self.max_ref_time = max_ref_time
        self.max_ref_num = max_ref_num

        self.n_repeat = n_repeats
        self.expaned_length = len(self.data_list) * self.n_repeat
        self.return_dict = return_dict

        # determine number of speakers and assign speaker id
        speakers = self.df[1].unique()
        self.speaker_dict = {speaker: i for i, speaker in enumerate(speakers)}

        # read duration

    def __len__(self):
        return self.expaned_length

    def __getitem__(self, idx):
        idx = idx % len(self.data_list)
        data = self.data_list[idx]
        path = data[0]

        wave, text, speaker_id, lang_token = self._load_tensor(data)

        speaker_label = torch.LongTensor([self.speaker_dict[speaker_id]])

        mel_tensor = preprocess(wave).squeeze()

        mel_len = int(np.ceil(wave.shape[0] / SPECT_PARAMS["hop_length"] * 1.5))
        mel_tensor = mel_tensor[:, :mel_len]

        acoustic_feature = mel_tensor.squeeze()

        return speaker_id, speaker_label, acoustic_feature, text, lang_token, path, wave

    def _load_tensor(self, data):
        wave_path, speaker_id, lang_id, text_norm, phonemes = data
        wave, sr = sf.read(osp.join(self.root_path, wave_path))
        if wave.shape[-1] == 2:
            wave = wave[:, 0].squeeze()
        if sr != 16000:
            wave = librosa.resample(wave, orig_sr=sr, target_sr=16000)
            # print(wave_path, sr)

        # wave = np.concatenate([np.zeros([5000]), wave, np.zeros([5000])], axis=0)

        text = phonemes.replace("\n", "")
        text = self.text_cleaner(text)

        text = torch.LongTensor(text)

        lang_token = torch.LongTensor([lang_dict[lang_id]])

        return wave, text, speaker_id, lang_token


def collate(batch):
    # batch[0] = wave, mel, text, f0, speakerid
    batch_size = len(batch)

    # sort by mel length
    lengths = [b[2].shape[1] for b in batch]
    batch_indexes = np.argsort(lengths)[::-1]
    batch = [batch[bid] for bid in batch_indexes]

    nmels = batch[0][2].size(0)
    max_mel_length = max([b[2].shape[1] for b in batch])
    max_text_length = max([b[3].shape[0] for b in batch])

    labels = [None for _ in range(batch_size)]
    mels = torch.zeros((batch_size, nmels, max_mel_length)).float() - 10
    texts = torch.zeros((batch_size, max_text_length)).long()
    # ref_texts = torch.zeros((batch_size, max_rtext_length)).long()

    input_lengths = torch.zeros(batch_size).long()
    output_lengths = torch.zeros(batch_size).long()
    paths = ['' for _ in range(batch_size)]
    waves = [None for _ in range(batch_size)]

    speaker_labels = torch.zeros(batch_size).long()
    lang_tokens = torch.zeros(batch_size).long()

    for bid, (speaker_id, label, mel, text, lang_token, path, wave) in enumerate(batch):
        mel_size = mel.size(1)
        text_size = text.size(0)
        labels[bid] = label
        mels[bid, :, :mel_size] = mel
        texts[bid, :text_size] = text
        input_lengths[bid] = text_size
        output_lengths[bid] = mel_size
        paths[bid] = path
        waves[bid] = wave
        speaker_labels[bid] = label
        lang_tokens[bid] = lang_token

    return waves, texts, input_lengths, mels, output_lengths, speaker_labels, lang_tokens, paths


def build_dataloader(list_path,
                     root_path,
                     validation=False,
                     min_length=50,
                     batch_size=4,
                     batch_length=None,
                     num_workers=1,
                     device='cpu',
                     collate_config={},
                     dataset_config={},
                     rank=0,
                     world_size=1,
                     n_repeats=1,
                     return_dict=False,
                     ):
    dataset = FilePathDataset(list_path, root_path, min_length=min_length, validation=validation,
                              n_repeats=n_repeats, return_dict=return_dict, **dataset_config)
    collate_fn = collate
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             # sampler=sampler,
                             num_workers=num_workers,
                             drop_last=True,
                             collate_fn=collate_fn,
                             pin_memory=True,
                             shuffle=True,
                             )

    return data_loader

