from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.nn import Linear, Module
from torch.nn import functional as F
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear
from torch.nn.parameter import Parameter
import math

from .rotary_embedding import RotaryEmbedding

rotary_emb = None

def multi_head_attention_forward(
        x,
        ipw,
        ipb,
        opw,
        opb,
        n_head,
        attn_mask,
        dropout=0.0,
        past_kv=None,
        use_cache=False,
        use_rope=False,
        rope=None,
):
    rotary_emb = rope
    B, T, C = x.size()

    q, k, v = torch._C._nn.linear(x, ipw, ipb).chunk(3, dim=-1)
    k = k.view(B, T, n_head, C // n_head).transpose(1, 2)  # (B, nh, T, hs)
    q = q.view(B, T, n_head, C // n_head).transpose(1, 2)  # (B, nh, T, hs)
    v = v.view(B, T, n_head, C // n_head).transpose(1, 2)  # (B, nh, T, hs)

    # implement RoPE here
    if use_rope:
        if rotary_emb is None:
            rotary_emb = RotaryEmbedding(dim = C // n_head)
            rotary_emb.to(x.device)
        if past_kv is None:
            try:
                q = rotary_emb.rotate_queries_or_keys(q)
                k = rotary_emb.rotate_queries_or_keys(k)
            except:
                print("?")
        else:
            q = rotary_emb.rotate_queries_or_keys(q, offset=past_kv[0].shape[-2])
            k = rotary_emb.rotate_queries_or_keys(k, offset=past_kv[0].shape[-2])
    if past_kv is not None:
        past_key = past_kv[0]
        past_value = past_kv[1]
        k = torch.cat((past_key, k), dim=-2)
        v = torch.cat((past_value, v), dim=-2)

    FULL_T = k.shape[-2]

    if use_cache is True:
        present = [k, v]
    else:
        present = None

    if T == 1 or attn_mask is None:
        with torch.backends.cuda.sdp_kernel():
            y = F.scaled_dot_product_attention(q, k, v)
    else:
        with torch.backends.cuda.sdp_kernel():
            if attn_mask.dtype == torch.bool:
                y = F.scaled_dot_product_attention(q, k, v, attn_mask=~attn_mask[:, :, FULL_T - T:FULL_T, :FULL_T], dropout_p=dropout)
            else:
                y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask[:, :, FULL_T - T:FULL_T, :FULL_T], dropout_p=dropout)

    y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side
    y = torch._C._nn.linear(y, opw, opb)
    return (y, present)

def multi_head_cross_attention_forward(
        q,
        k,
        v,
        ipw,
        ipb,
        opw,
        opb,
        n_head,
        attn_mask,
        dropout=0.0,
        past_kv=None,
        use_cache=False,
        use_rope=False,
        rope=None,
):
    rotary_emb = rope
    B, qT, C = q.size()
    _, kT, _ = k.size()
    _, vT, _ = v.size()

    q = torch._C._nn.linear(q, ipw[:C, :], ipb[:C])
    k = torch._C._nn.linear(k, ipw[C:2 * C, :], ipb[C:2 * C])
    v = torch._C._nn.linear(v, ipw[2 * C:, :], ipb[2 * C:])
    q = q.view(B, qT, n_head, C // n_head).transpose(1, 2)  # (B, nh, T, hs)
    k = k.view(B, kT, n_head, C // n_head).transpose(1, 2)  # (B, nh, T, hs)
    v = v.view(B, vT, n_head, C // n_head).transpose(1, 2)  # (B, nh, T, hs)

    # implement RoPE here
    if use_rope:
        if rotary_emb is None:
            rotary_emb = RotaryEmbedding(dim = C // n_head)
            rotary_emb.to(q.device)
        if past_kv is None:
            q = rotary_emb.rotate_queries_or_keys(q)
            k = rotary_emb.rotate_queries_or_keys(k)
        else:
            q = rotary_emb.rotate_queries_or_keys(q, offset=past_kv[0].shape[-2])
            k = rotary_emb.rotate_queries_or_keys(k, offset=past_kv[0].shape[-2])
    else:
        q, k = q.contiguous(), k.contiguous()
    if past_kv is not None:
        past_key = past_kv[0]
        past_value = past_kv[1]
        k = torch.cat((past_key, k), dim=-2)
        v = torch.cat((past_value, v), dim=-2)

    q_FULL_T = q.shape[-2]
    k_FULL_T = k.shape[-2]

    if use_cache is True:
        present = [k, v]
    else:
        present = None

    if qT == 1 or attn_mask is None:
        with torch.backends.cuda.sdp_kernel():
            y = F.scaled_dot_product_attention(q, k, v)
    else:
        with torch.backends.cuda.sdp_kernel():
            if attn_mask.dtype == torch.bool:
                y = F.scaled_dot_product_attention(q, k, v, attn_mask=~attn_mask[q_FULL_T - qT:q_FULL_T, :k_FULL_T], dropout_p=dropout)
            else:
                y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask[q_FULL_T - qT:q_FULL_T, :k_FULL_T], dropout_p=dropout)

    # calculate and return attention weights
    attn_map = torch.bmm(q.view(-1, qT, C // n_head), k.view(-1, kT, C // n_head).transpose(1, 2)).view(B, n_head, qT, kT)
    attn_map = attn_map / math.sqrt(C // n_head)
    if attn_mask.dtype == torch.bool:
        attn_map = attn_map.masked_fill(attn_mask[q_FULL_T - qT:q_FULL_T, :k_FULL_T], -1e5)
    else:
        attn_map += attn_mask[q_FULL_T - qT:q_FULL_T, :k_FULL_T].unsqueeze(1)
    attn_map = F.softmax(attn_map.mean(dim=1), dim=-1)
    y = y.transpose(1, 2).contiguous().view(B, qT, C)  # re-assemble all head outputs side by side
    y = torch._C._nn.linear(y, opw, opb)
    return (y, present), attn_map


class MultiheadAttention(Module):
    r"""Allows the model to jointly attend to information
    from different representation subspaces as described in the paper:
    `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_.

    Multi-Head Attention is defined as:

    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O

    where :math:`head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)`.

    ``forward()`` will use a special optimized implementation if all of the following
    conditions are met:

    - self attention is being computed (i.e., ``query``, ``key``, and ``value`` are the same tensor. This
      restriction will be loosened in the future.)
    - Either autograd is disabled (using ``torch.inference_mode`` or ``torch.no_grad``) or no tensor argument ``requires_grad``
    - training is disabled (using ``.eval()``)
    - dropout is 0
    - ``add_bias_kv`` is ``False``
    - ``add_zero_attn`` is ``False``
    - ``batch_first`` is ``True`` and the input is batched
    - ``kdim`` and ``vdim`` are equal to ``embed_dim``
    - at most one of ``key_padding_mask`` or ``attn_mask`` is passed
    - if a `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_ is passed, neither ``key_padding_mask``
      nor ``attn_mask`` is passed

    If the optimized implementation is in use, a
    `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_ can be passed for
    ``query``/``key``/``value`` to represent padding more efficiently than using a
    padding mask. In this case, a `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_
    will be returned, and an additional speedup proportional to the fraction of the input
    that is padding can be expected.

    Args:
        embed_dim: Total dimension of the model.
        num_heads: Number of parallel attention heads. Note that ``embed_dim`` will be split
            across ``num_heads`` (i.e. each head will have dimension ``embed_dim // num_heads``).
        dropout: Dropout probability on ``attn_output_weights``. Default: ``0.0`` (no dropout).
        bias: If specified, adds bias to input / output projection layers. Default: ``True``.
        add_bias_kv: If specified, adds bias to the key and value sequences at dim=0. Default: ``False``.
        add_zero_attn: If specified, adds a new batch of zeros to the key and value sequences at dim=1.
            Default: ``False``.
        kdim: Total number of features for keys. Default: ``None`` (uses ``kdim=embed_dim``).
        vdim: Total number of features for values. Default: ``None`` (uses ``vdim=embed_dim``).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).

    Examples::

        >>> # xdoctest: +SKIP
        >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)

    """
    __constants__ = ["batch_first"]
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        kdim=None,
        vdim=None,
        batch_first=False,
        linear1_cls=Linear,
        linear2_cls=Linear,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = (
            self.kdim == embed_dim and self.vdim == embed_dim
        )

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        if add_bias_kv:
            self.bias_k = Parameter(
                torch.empty((1, 1, embed_dim), **factory_kwargs)
            )
            self.bias_v = Parameter(
                torch.empty((1, 1, embed_dim), **factory_kwargs)
            )
        else:
            self.bias_k = self.bias_v = None

        if linear1_cls == Linear:
            if not self._qkv_same_embed_dim:
                self.q_proj_weight = Parameter(
                    torch.empty((embed_dim, embed_dim), **factory_kwargs)
                )
                self.k_proj_weight = Parameter(
                    torch.empty((embed_dim, self.kdim), **factory_kwargs)
                )
                self.v_proj_weight = Parameter(
                    torch.empty((embed_dim, self.vdim), **factory_kwargs)
                )
                self.register_parameter("in_proj_weight", None)
            else:
                self.in_proj_weight = Parameter(
                    torch.empty((3 * embed_dim, embed_dim), **factory_kwargs)
                )
                self.register_parameter("q_proj_weight", None)
                self.register_parameter("k_proj_weight", None)
                self.register_parameter("v_proj_weight", None)

            if bias:
                self.in_proj_bias = Parameter(
                    torch.empty(3 * embed_dim, **factory_kwargs)
                )
            else:
                self.register_parameter("in_proj_bias", None)
            self.out_proj = NonDynamicallyQuantizableLinear(
                embed_dim, embed_dim, bias=bias, **factory_kwargs
            )

            self._reset_parameters()
        else:
            if not self._qkv_same_embed_dim:
                raise NotImplementedError
            else:
                self.in_proj_linear = linear1_cls(
                    embed_dim, 3 * embed_dim, bias=bias, **factory_kwargs
                )
                self.in_proj_weight = self.in_proj_linear.weight

                self.register_parameter("q_proj_weight", None)
                self.register_parameter("k_proj_weight", None)
                self.register_parameter("v_proj_weight", None)

                if bias:
                    self.in_proj_bias = self.in_proj_linear.bias
                else:
                    self.register_parameter("in_proj_bias", None)

            self.out_proj = linear2_cls(
                embed_dim, embed_dim, bias=bias, **factory_kwargs
            )

            if self.bias_k is not None:
                xavier_normal_(self.bias_k)
            if self.bias_v is not None:
                xavier_normal_(self.bias_v)

        self.add_zero_attn = add_zero_attn

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.0)
            constant_(self.out_proj.bias, 0.0)

        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def __setstate__(self, state):
        # Support loading old MultiheadAttention checkpoints generated by v1.1.0
        if "_qkv_same_embed_dim" not in state:
            state["_qkv_same_embed_dim"] = True

        super(MultiheadAttention, self).__setstate__(state)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        average_attn_weights: bool = True,
        use_rope: bool = False,
        rope = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        r"""
        Args:
            query: Query embeddings of shape :math:`(L, E_q)` for unbatched input, :math:`(L, N, E_q)` when ``batch_first=False``
                or :math:`(N, L, E_q)` when ``batch_first=True``, where :math:`L` is the target sequence length,
                :math:`N` is the batch size, and :math:`E_q` is the query embedding dimension ``embed_dim``.
                Queries are compared against key-value pairs to produce the output.
                See "Attention Is All You Need" for more details.
            key: Key embeddings of shape :math:`(S, E_k)` for unbatched input, :math:`(S, N, E_k)` when ``batch_first=False``
                or :math:`(N, S, E_k)` when ``batch_first=True``, where :math:`S` is the source sequence length,
                :math:`N` is the batch size, and :math:`E_k` is the key embedding dimension ``kdim``.
                See "Attention Is All You Need" for more details.
            value: Value embeddings of shape :math:`(S, E_v)` for unbatched input, :math:`(S, N, E_v)` when
                ``batch_first=False`` or :math:`(N, S, E_v)` when ``batch_first=True``, where :math:`S` is the source
                sequence length, :math:`N` is the batch size, and :math:`E_v` is the value embedding dimension ``vdim``.
                See "Attention Is All You Need" for more details.
            key_padding_mask: If specified, a mask of shape :math:`(N, S)` indicating which elements within ``key``
                to ignore for the purpose of attention (i.e. treat as "padding"). For unbatched `query`, shape should be :math:`(S)`.
                Binary and byte masks are supported.
                For a binary mask, a ``True`` value indicates that the corresponding ``key`` value will be ignored for
                the purpose of attention. For a float mask, it will be directly added to the corresponding ``key`` value.
            need_weights: If specified, returns ``attn_output_weights`` in addition to ``attn_outputs``.
                Default: ``True``.
            attn_mask: If specified, a 2D or 3D mask preventing attention to certain positions. Must be of shape
                :math:`(L, S)` or :math:`(N\cdot\text{num\_heads}, L, S)`, where :math:`N` is the batch size,
                :math:`L` is the target sequence length, and :math:`S` is the source sequence length. A 2D mask will be
                broadcasted across the batch while a 3D mask allows for a different mask for each entry in the batch.
                Binary, byte, and float masks are supported. For a binary mask, a ``True`` value indicates that the
                corresponding position is not allowed to attend. For a byte mask, a non-zero value indicates that the
                corresponding position is not allowed to attend. For a float mask, the mask values will be added to
                the attention weight.
            average_attn_weights: If true, indicates that the returned ``attn_weights`` should be averaged across
                heads. Otherwise, ``attn_weights`` are provided separately per head. Note that this flag only has an
                effect when ``need_weights=True``. Default: ``True`` (i.e. average weights across heads)

        Outputs:
            - **attn_output** - Attention outputs of shape :math:`(L, E)` when input is unbatched,
              :math:`(L, N, E)` when ``batch_first=False`` or :math:`(N, L, E)` when ``batch_first=True``,
              where :math:`L` is the target sequence length, :math:`N` is the batch size, and :math:`E` is the
              embedding dimension ``embed_dim``.
            - **attn_output_weights** - Only returned when ``need_weights=True``. If ``average_attn_weights=True``,
              returns attention weights averaged across heads of shape :math:`(L, S)` when input is unbatched or
              :math:`(N, L, S)`, where :math:`N` is the batch size, :math:`L` is the target sequence length, and
              :math:`S` is the source sequence length. If ``average_attn_weights=False``, returns attention weights per
              head of shape :math:`(\text{num\_heads}, L, S)` when input is unbatched or :math:`(N, \text{num\_heads}, L, S)`.

            .. note::
                `batch_first` argument is ignored for unbatched inputs.
        """
        is_batched = query.dim() == 3
        if key_padding_mask is not None:
            _kpm_dtype = key_padding_mask.dtype
            if _kpm_dtype != torch.bool and not torch.is_floating_point(
                key_padding_mask
            ):
                raise AssertionError(
                    "only bool and floating types of key_padding_mask are supported"
                )
        why_not_fast_path = ""
        if not is_batched:
            why_not_fast_path = f"input not batched; expected query.dim() of 3 but got {query.dim()}"
        elif query is not key or key is not value:
            # When lifting this restriction, don't forget to either
            # enforce that the dtypes all match or test cases where
            # they don't!
            why_not_fast_path = "non-self attention was used (query, key, and value are not the same Tensor)"
        elif (
            self.in_proj_bias is not None
            and query.dtype != self.in_proj_bias.dtype
        ):
            why_not_fast_path = f"dtypes of query ({query.dtype}) and self.in_proj_bias ({self.in_proj_bias.dtype}) don't match"
        elif (
            self.in_proj_weight is not None
            and query.dtype != self.in_proj_weight.dtype
        ):
            # this case will fail anyway, but at least they'll get a useful error message.
            why_not_fast_path = f"dtypes of query ({query.dtype}) and self.in_proj_weight ({self.in_proj_weight.dtype}) don't match"
        elif self.training:
            why_not_fast_path = "training is enabled"
        elif not self.batch_first:
            why_not_fast_path = "batch_first was not True"
        elif self.bias_k is not None:
            why_not_fast_path = "self.bias_k was not None"
        elif self.bias_v is not None:
            why_not_fast_path = "self.bias_v was not None"
        elif self.dropout:
            why_not_fast_path = f"dropout was {self.dropout}, required zero"
        elif self.add_zero_attn:
            why_not_fast_path = "add_zero_attn was enabled"
        elif not self._qkv_same_embed_dim:
            why_not_fast_path = "_qkv_same_embed_dim was not True"
        elif attn_mask is not None:
            why_not_fast_path = "attn_mask was not None"
        elif query.is_nested and key_padding_mask is not None:
            why_not_fast_path = (
                "key_padding_mask is not supported with NestedTensor input"
            )
        elif self.num_heads % 2 == 1:
            why_not_fast_path = "num_heads is odd"
        elif torch.is_autocast_enabled():
            why_not_fast_path = "autocast is enabled"

        if not why_not_fast_path:
            tensor_args = (
                query,
                key,
                value,
                self.in_proj_weight,
                self.in_proj_bias,
                self.out_proj.weight,
                self.out_proj.bias,
            )
            # We have to use list comprehensions below because TorchScript does not support
            # generator expressions.
            if torch.overrides.has_torch_function(tensor_args):
                why_not_fast_path = "some Tensor argument has_torch_function"
            elif not all(
                [
                    (x is None or x.is_cuda or "cpu" in str(x.device))
                    for x in tensor_args
                ]
            ):
                why_not_fast_path = (
                    "some Tensor argument is neither CUDA nor CPU"
                )
            elif torch.is_grad_enabled() and any(
                [x is not None and x.requires_grad for x in tensor_args]
            ):
                why_not_fast_path = (
                    "grad is enabled and at least one of query or the "
                    "input/output projection weights or biases requires_grad"
                )
            if not why_not_fast_path:
                return torch._native_multi_head_attention(
                    query,
                    key,
                    value,
                    self.embed_dim,
                    self.num_heads,
                    self.in_proj_weight,
                    self.in_proj_bias,
                    self.out_proj.weight,
                    self.out_proj.bias,
                    key_padding_mask
                    if key_padding_mask is not None
                    else attn_mask,
                    need_weights,
                    average_attn_weights,
                    1
                    if key_padding_mask is not None
                    else 0
                    if attn_mask is not None
                    else None,
                )

        any_nested = query.is_nested or key.is_nested or value.is_nested
        assert not any_nested, (
            "MultiheadAttention does not support NestedTensor outside of its fast path. "
            + f"The fast path was not hit because {why_not_fast_path}"
        )
        x = query
        if query.shape[1] == key.shape[1]:
            attn_output, _ = multi_head_attention_forward(
                x=x,
                ipw=self.in_proj_weight,
                ipb=self.in_proj_bias,
                opw=self.out_proj.weight,
                opb=self.out_proj.bias,
                n_head=self.num_heads,
                attn_mask=attn_mask,
                dropout=self.dropout,
                past_kv=None,
                use_cache=False,
                use_rope=use_rope,
                rope=rope,
            )
        else:
            attn_output = multi_head_cross_attention_forward(
                q=query,
                k=key,
                v=value,
                ipw=self.in_proj_weight,
                ipb=self.in_proj_bias,
                opw=self.out_proj.weight,
                opb=self.out_proj.bias,
                n_head=self.num_heads,
                attn_mask=attn_mask,
                dropout=self.dropout,
                past_kv=None,
                use_cache=False,
                use_rope=use_rope,
                rope=rope,
            )
        return attn_output, None

    def infer(self,
              x: Tensor,
              key_padding_mask: Optional[Tensor] = None,
              need_weights: bool = True,
              attn_mask: Optional[Tensor] = None,
              average_attn_weights: bool = True,
              past_kv = None,
              use_cache = False,
              use_rope = False,
              rope = None
              ):
        # x = x.transpose(1, 0)
        y, kv = multi_head_attention_forward(
                x=x,
                ipw=self.in_proj_weight,
                ipb=self.in_proj_bias,
                opw=self.out_proj.weight,
                opb=self.out_proj.bias,
                n_head=self.num_heads,
                attn_mask=attn_mask,
                past_kv=past_kv,
                use_cache=use_cache,
                use_rope=use_rope,
                rope=rope,
        )
        return (y, kv)

    def __repr__(self):
        s = (
            f"MultiheadAttention("
            f"embed_dim={self.embed_dim}, "
            f"num_heads={self.num_heads}, "
            f"dropout={self.dropout}, "
            f"bias={self.bias_k is not None}, "
            f"add_bias_kv={self.bias_k is not None}, "
            f"add_zero_attn={self.add_zero_attn}, "
            f"kdim={self.kdim}, "
            f"vdim={self.vdim}, "
            f"batch_first={self.batch_first}"
        )
        if self._qkv_same_embed_dim:
            s += ", linear1_cls=Linear"
        else:
            s += ", linear1_cls=NotImplemented"
        if self.bias_k is not None:
            s += ", bias_k=Parameter containing:\n" + str(self.bias_k)
        if self.bias_v is not None:
            s += ", bias_v=Parameter containing:\n" + str(self.bias_v)
        s += ")"
        return s
