import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from munch import Munch
import json

class AttrDict(dict):
  def __init__(self, *args, **kwargs):
    super(AttrDict, self).__init__(*args, **kwargs)
    self.__dict__ = self

def init_weights(m, mean=0.0, std=0.01):
  classname = m.__class__.__name__
  if classname.find("Conv") != -1:
    m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
  return int((kernel_size*dilation - dilation)/2)


def convert_pad_shape(pad_shape):
  l = pad_shape[::-1]
  pad_shape = [item for sublist in l for item in sublist]
  return pad_shape


def intersperse(lst, item):
  result = [item] * (len(lst) * 2 + 1)
  result[1::2] = lst
  return result


def kl_divergence(m_p, logs_p, m_q, logs_q):
  """KL(P||Q)"""
  kl = (logs_q - logs_p) - 0.5
  kl += 0.5 * (torch.exp(2. * logs_p) + ((m_p - m_q)**2)) * torch.exp(-2. * logs_q)
  return kl


def rand_gumbel(shape):
  """Sample from the Gumbel distribution, protect from overflows."""
  uniform_samples = torch.rand(shape) * 0.99998 + 0.00001
  return -torch.log(-torch.log(uniform_samples))


def rand_gumbel_like(x):
  g = rand_gumbel(x.size()).to(dtype=x.dtype, device=x.device)
  return g


def slice_segments(x, ids_str, segment_size=4):
  ret = torch.zeros_like(x[:, :, :segment_size])
  for i in range(x.size(0)):
    idx_str = ids_str[i]
    idx_end = idx_str + segment_size
    ret[i] = x[i, :, idx_str:idx_end]
  return ret

def slice_segments_audio(x, ids_str, segment_size=4):
  ret = torch.zeros_like(x[:, :segment_size])
  for i in range(x.size(0)):
    idx_str = ids_str[i]
    idx_end = idx_str + segment_size
    ret[i] = x[i, idx_str:idx_end]
  return ret

def rand_slice_segments(x, x_lengths=None, segment_size=4):
  b, d, t = x.size()
  if x_lengths is None:
    x_lengths = t
  ids_str_max = x_lengths - segment_size + 1
  ids_str = ((torch.rand([b]).to(device=x.device) * ids_str_max).clip(0)).to(dtype=torch.long)
  ret = slice_segments(x, ids_str, segment_size)
  return ret, ids_str


def get_timing_signal_1d(
    length, channels, min_timescale=1.0, max_timescale=1.0e4):
  position = torch.arange(length, dtype=torch.float)
  num_timescales = channels // 2
  log_timescale_increment = (
      math.log(float(max_timescale) / float(min_timescale)) /
      (num_timescales - 1))
  inv_timescales = min_timescale * torch.exp(
      torch.arange(num_timescales, dtype=torch.float) * -log_timescale_increment)
  scaled_time = position.unsqueeze(0) * inv_timescales.unsqueeze(1)
  signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], 0)
  signal = F.pad(signal, [0, 0, 0, channels % 2])
  signal = signal.view(1, channels, length)
  return signal


def add_timing_signal_1d(x, min_timescale=1.0, max_timescale=1.0e4):
  b, channels, length = x.size()
  signal = get_timing_signal_1d(length, channels, min_timescale, max_timescale)
  return x + signal.to(dtype=x.dtype, device=x.device)


def cat_timing_signal_1d(x, min_timescale=1.0, max_timescale=1.0e4, axis=1):
  b, channels, length = x.size()
  signal = get_timing_signal_1d(length, channels, min_timescale, max_timescale)
  return torch.cat([x, signal.to(dtype=x.dtype, device=x.device)], axis)


def subsequent_mask(length):
  mask = torch.tril(torch.ones(length, length)).unsqueeze(0).unsqueeze(0)
  return mask


@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
  n_channels_int = n_channels[0]
  in_act = input_a + input_b
  t_act = torch.tanh(in_act[:, :n_channels_int, :])
  s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
  acts = t_act * s_act
  return acts


def convert_pad_shape(pad_shape):
  l = pad_shape[::-1]
  pad_shape = [item for sublist in l for item in sublist]
  return pad_shape


def shift_1d(x):
  x = F.pad(x, convert_pad_shape([[0, 0], [0, 0], [1, 0]]))[:, :, :-1]
  return x


def sequence_mask(length, max_length=None):
  if max_length is None:
    max_length = length.max()
  x = torch.arange(max_length, dtype=length.dtype, device=length.device)
  return x.unsqueeze(0) < length.unsqueeze(1)


def generate_path(duration, mask):
  """
  duration: [b, 1, t_x]
  mask: [b, 1, t_y, t_x]
  """
  device = duration.device
  
  b, _, t_y, t_x = mask.shape
  cum_duration = torch.cumsum(duration, -1)
  
  cum_duration_flat = cum_duration.view(b * t_x)
  path = sequence_mask(cum_duration_flat, t_y).to(mask.dtype)
  path = path.view(b, t_x, t_y)
  path = path - F.pad(path, convert_pad_shape([[0, 0], [1, 0], [0, 0]]))[:, :-1]
  path = path.unsqueeze(1).transpose(2,3) * mask
  return path


def clip_grad_value_(parameters, clip_value, norm_type=2):
  if isinstance(parameters, torch.Tensor):
    parameters = [parameters]
  parameters = list(filter(lambda p: p.grad is not None, parameters))
  norm_type = float(norm_type)
  if clip_value is not None:
    clip_value = float(clip_value)

  total_norm = 0
  for p in parameters:
    param_norm = p.grad.data.norm(norm_type)
    total_norm += param_norm.item() ** norm_type
    if clip_value is not None:
      p.grad.data.clamp_(min=-clip_value, max=clip_value)
  total_norm = total_norm ** (1. / norm_type)
  return total_norm

def log_norm(x, mean=-4, std=4, dim=2):
  """
  normalized log mel -> mel -> norm -> log(norm)
  """
  x = torch.log(torch.exp(x * std + mean).norm(dim=dim))
  return x

def load_F0_models(path):
  # load F0 model
  from .JDC.model import JDCNet
  F0_model = JDCNet(num_class=1, seq_len=192)
  params = torch.load(path, map_location='cpu')['net']
  F0_model.load_state_dict(params)
  _ = F0_model.train()

  return F0_model


def build_model(args):
  from facodec import FACodecEncoderV2, FACodecDecoderV2, CNNLSTM, ArcMarginProduct
  from .phoneme_predictor import PhonemePredictor
  from gradient_reversal import GradientReversal
  from .msstftd import MultiScaleSTFTDiscriminator
  from .discriminators import MultiResolutionDiscriminator, MultiPeriodDiscriminator
  json_config = json.loads(open("./configs/bigvgan_base_24khz.json").read())
  h = AttrDict(json_config)

  fa_encoder = FACodecEncoderV2(
    ngf=32,
    up_ratios=[2, 4, 5, 5],
    out_channels=256,
  )

  fa_decoder = FACodecDecoderV2(
    in_channels=256,
    upsample_initial_channel=1024,
    ngf=32,
    up_ratios=[5, 5, 4, 2],
    vq_num_q_c=2,
    vq_num_q_p=1,
    vq_num_q_r=3,
    vq_dim=256,
    codebook_dim=8,
    codebook_size_prosody=10,
    codebook_size_content=10,
    codebook_size_residual=10,
    use_gr_x_timbre=False,
    use_gr_residual_f0=False,
    use_gr_residual_phone=False,
    use_gr_content_f0=False,
    use_gr_prosody_phone=False,
  )

  # content_phoneme_predictor = PhonemePredictor(
  #   d_model=256,
  #   nhead=4,
  #   num_layers=6,
  #   n_token=230,
  #   n_lang=1,
  # )
  #
  # prosody_phoneme_predictor = nn.Sequential(
  #   GradientReversal(alpha=1.0),
  #   PhonemePredictor(
  #     d_model=256,
  #     nhead=4,
  #     num_layers=6,
  #     n_token=230,
  #     n_lang=1,
  #   )
  # )
  #
  # res_phoneme_predictor = nn.Sequential(
  #   GradientReversal(alpha=1.0),
  #   PhonemePredictor(
  #     d_model=256,
  #     nhead=4,
  #     num_layers=6,
  #     n_token=230,
  #     n_lang=1,
  #   )
  # )
  #
  # prosody_f0n_predictor = CNNLSTM(indim=256, outdim=1, head=2)
  #
  # content_f0n_predictor = nn.Sequential(
  #   GradientReversal(alpha=1.0),
  #   CNNLSTM(indim=256, outdim=1, head=2)
  # )
  #
  # res_f0n_predictor = nn.Sequential(
  #   GradientReversal(alpha=1.0),
  #   CNNLSTM(indim=256, outdim=1, head=2)
  # )
  #
  # timbre_predictor = ArcMarginProduct(256, 114514, s=30, m=0.5)
  #
  # x_timbre_encoder = nn.Sequential(
  #   GradientReversal(alpha=1),
  #   CNNLSTM(256, 256, 1, global_pred=True),
  # )
  #
  # x_timbre_predictor = ArcMarginProduct(256, 114514, s=30, m=0.5)

  mrd = MultiResolutionDiscriminator(h)
  mpd = MultiPeriodDiscriminator(h)
  # stft_disc = MultiScaleSTFTDiscriminator(filters=32)

  nets = Munch(
    fa_encoder=fa_encoder,
    fa_decoder=fa_decoder,
    # content_phoneme_predictor=content_phoneme_predictor,
    # prosody_phoneme_predictor=prosody_phoneme_predictor,
    # res_phoneme_predictor=res_phoneme_predictor,
    # prosody_f0n_predictor=prosody_f0n_predictor,
    # content_f0n_predictor=content_f0n_predictor,
    # res_f0n_predictor=res_f0n_predictor,
    # timbre_predictor=timbre_predictor,
    # x_timbre_encoder=x_timbre_encoder,
    # x_timbre_predictor=x_timbre_predictor,
    mrd=mrd,
    mpd=mpd,
    # stft_disc=stft_disc,
  )

  return nets


def load_checkpoint(model, optimizer, path, load_only_params=True, ignore_modules=[]):
  state = torch.load(path, map_location='cpu')
  params = state['net']
  for key in model:
    if key in params and key not in ignore_modules:
      print('%s loaded' % key)
      model[key].load_state_dict(params[key], strict=True)
  _ = [model[key].eval() for key in model]

  if not load_only_params:
    epoch = state["epoch"] + 1
    iters = state["iters"]
    optimizer.load_state_dict(state["optimizer"])
  else:
    epoch = state["epoch"] + 1
    iters = state["iters"]

  return model, optimizer, epoch, iters

def recursive_munch(d):
  if isinstance(d, dict):
    return Munch((k, recursive_munch(v)) for k, v in d.items())
  elif isinstance(d, list):
    return [recursive_munch(v) for v in d]
  else:
    return d