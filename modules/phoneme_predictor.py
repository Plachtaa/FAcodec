import math
import torch
from torch import nn
from torch.nn import TransformerEncoder
import torch.nn.functional as F
from modules.layers import MFCC, Attention, LinearNorm, ConvNorm, ConvBlock

from transformer_modules.transformer import TransformerEncoder, TransformerEncoderLayer, LayerNorm
from transformer_modules.transformer import TransformerDecoder, TransformerDecoderLayer

def build_model(model_params={}, model_type='asr'):
    d_model = model_params.get('hidden_dim', 512)
    nhead = model_params.get('nhead', 8)
    num_layers = model_params.get('num_layers', 6)
    n_lang = model_params.get('n_lang', 6)
    n_token = model_params.get('n_token', 183)
    model = PhonemePredictor(d_model=d_model, nhead=nhead, num_layers=num_layers, n_token=n_token, n_lang=n_lang)
    return model

norm_first = True

class PhonemePredictor(torch.nn.Module):
    def __init__(self, d_model=512, nhead=8, num_layers=6, n_token=178, n_lang=6):
        super(PhonemePredictor, self).__init__()
        self.n_token = n_token

        self.mel_encoder = TransformerEncoder(
            TransformerEncoderLayer(
                d_model,
                nhead,
                dim_feedforward=d_model * 4,
                dropout=0.1,
                batch_first=True,
                norm_first=norm_first,
            ),
            num_layers=num_layers,
            norm=LayerNorm(d_model) if norm_first else None,
        )
        self.asr_decoder = TransformerDecoder(
            TransformerDecoderLayer(
                d_model,
                nhead,
                dim_feedforward=d_model * 4,
                dropout=0.1,
                batch_first=True,
                norm_first=norm_first,
            ),
            num_layers=num_layers,
            norm=LayerNorm(d_model) if norm_first else None,
        )

        self.text_embed = nn.Embedding(n_token + 1, d_model) # plus bos token
        self.lang_embed = nn.Embedding(n_lang, d_model)
        self.bos_idx = n_token
        self.audio_prenet = ConvNorm(256, d_model, kernel_size=7, padding=3, stride=1)
        self.nhead = nhead
        self.text_predictor = nn.Linear(d_model, n_token)

        self.ctc_linear = nn.Sequential(
            LinearNorm(d_model, d_model),
            nn.ReLU(),
            LinearNorm(d_model, n_token))

    def forward(self, mels, texts, langs=None, mel_lens=None, text_lens=None,):
        """Attention mechanism for radtts. Unlike in Flowtron, we have no
        restrictions such as causality etc, since we only need this during
        training.

        Args:
            mels (torch.tensor): B x C x T1 tensor (likely mel data)
            texts (torch.tensor): B x C2 x T2 tensor (text data)
            langs (torch.tensor): B x 1 tensor (language id)
            mel_lens: lengths for sorting the queries in descending order
            text_lens: lengths for sorting the keys in descending order
        Output:
            attn (torch.tensor): B x 1 x T1 x T2 attention mask.
                                 Final dim T2 should sum to 1
        """
        B = mels.shape[0]
        # append bos token
        texts = torch.cat([torch.ones([B, 1], dtype=torch.int).to(texts.device) * self.bos_idx, texts], dim=1)
        text_lens = text_lens + 1

        texts = self.text_embed(texts)
        mels = self.audio_prenet(mels).transpose(1, 2)

        langs = self.lang_embed(langs)

        # append language embedding to mels
        mels = torch.cat([langs.unsqueeze(1), mels], dim=1)
        mel_lens = mel_lens + 1

        x_mask = torch.zeros([B, 1, texts.shape[1]]).to(texts.device)
        y_mask = torch.zeros([B, 1, mels.shape[1]]).to(mels.device)
        # fill masks with 1s up to the length of the sequence
        for i in range(B):
            x_mask[i, :, :text_lens[i]] = 1
            y_mask[i, :, :mel_lens[i]] = 1
        attn_mask = torch.unsqueeze(x_mask, -1) * torch.unsqueeze(y_mask, 2)

        x_mask, y_mask, attn_mask = ~x_mask.type(torch.bool), ~y_mask.type(torch.bool), ~attn_mask.type(torch.bool)
        x_mask = x_mask.repeat(1, x_mask.shape[2], 1)
        y_mask = y_mask.repeat(1, y_mask.shape[2], 1)
        x_mask = x_mask.unsqueeze(1).repeat(1, self.nhead, 1, 1)
        y_mask = y_mask.unsqueeze(1).repeat(1, self.nhead, 1, 1)
        encoder_hidden_states, _ = self.mel_encoder((mels, None), mask=y_mask, use_rope=True)
        # make causal mask for tgt_mask
        tgt_mask = torch.triu(torch.ones([texts.shape[1], texts.shape[1]]), diagonal=1).to(texts.device)
        tgt_mask = tgt_mask.view(1, 1, texts.shape[1], texts.shape[1]).repeat(B, self.nhead, 1, 1).type(torch.bool)
        decoder_hidden_states, s2s_attn = self.asr_decoder(tgt=texts,
                                                    memory=encoder_hidden_states,
                                                    tgt_mask=tgt_mask,
                                                    memory_mask=y_mask[..., 0, :].unsqueeze(2).repeat(1, 1, x_mask.shape[2], 1),
                                                    return_attn=True,
                                                    use_rope=True)
        s2s_pred = self.text_predictor(decoder_hidden_states)

        ctc_logit = self.ctc_linear(encoder_hidden_states[:, 1:, :])
        return ctc_logit, s2s_pred[:, :-1, :], s2s_attn[-1][:, :-1, 1:]
