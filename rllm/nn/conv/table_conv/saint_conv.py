from __future__ import annotations
from typing import Dict, List, Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn, einsum

from rllm.types import ColType
from rllm.nn.pre_encoder import FTTransformerPreEncoder


def _exists(val):
    return val is not None


def _default(val, d):
    return val if _exists(val) else d


class MLP(nn.Module):
    r"""Classical Multilayer Perceptron."""

    def __init__(self, dims, act=None):
        super().__init__()
        dims_pairs = list(zip(dims[:-1], dims[1:]))
        layers = []
        for ind, (dim_in, dim_out) in enumerate(dims_pairs):
            is_last = ind >= (len(dims_pairs) - 1)
            linear = nn.Linear(dim_in, dim_out)
            layers.append(linear)

            if is_last:
                continue

            act = _default(act, nn.ReLU())
            layers.append(act)

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

    def reset_parameters(self) -> None:
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()


class PreNorm(nn.Module):
    r"""Pre-Normalization before the main layer."""

    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

    def reset_parameters(self) -> None:
        if hasattr(self.fn, "reset_parameters"):
            self.fn.reset_parameters()


class GEGLU(nn.Module):
    r"""GEGLU activation proposed in the `"GLU Variants Improve Transformer"
    <https://arxiv.org/abs/2002.05202>`_ paper.
    """

    def forward(self, x: Tensor) -> Tensor:
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    r"""Feedforward network.

    Args:
        dim (int): Input channel dimensionality
        mult (int): Expansion factor of the first layer (default: :obj:`4`)
        dropout (float): Percentage of random deactivation (default: :obj:`0.`)
    """

    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
        )

    def forward(self, x, **kwargs):
        return self.net(x)

    def reset_parameters(self) -> None:
        self.net[0].reset_parameters()
        self.net[3].reset_parameters()


class SelfAttention(nn.Module):
    r"""Self-attention module.

    Args:
        dim (int): Input channel dimensionality
        heads (int): Number of heads in Attention module (default: :obj:`8.`)
        dim_head(int): Dimension of each attention head (default: :obj:`16.`)
        dropout (float): Percentage of random deactivation (default: :obj:`0.`)
    """

    def __init__(
        self,
        dim,
        heads=8,
        dim_head=16,
        dropout=0.0,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.dropout = nn.Dropout(dropout)

    def _rearrange_qkv(self, x: Tensor) -> Tensor:
        # reshape b n (h d) -> b h n d
        b, num_cols, dim = x.shape
        d_head = dim // self.heads
        x = x.reshape(b, num_cols, self.heads, d_head)
        x = x.permute(0, 2, 1, 3)
        return x

    def forward(self, x):
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q = self._rearrange_qkv(q)
        k = self._rearrange_qkv(k)
        v = self._rearrange_qkv(v)
        sim = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale

        attn = sim.softmax(dim=-1)
        dropped_attn = self.dropout(attn)

        out = einsum("b h i j, b h j d -> b h i d", dropped_attn, v)
        # reshape b h n d -> b n (h d)
        out = out.permute(0, 2, 1, 3)
        out = out.reshape(out.size(0), out.size(1), -1)
        return self.to_out(out), attn

    def reset_parameters(self) -> None:
        self.to_qkv.reset_parameters()
        self.to_out.reset_parameters()


class Transformer(nn.Module):
    r"""
    This Transformer refers to the Encoder part of the complete structure of
    the transformer paper introduced in the `"Attention Is All You Need"
    <https://arxiv.org/abs/1706.03762>`_ paper.

    Args:
        dim (int): Input channel dimensionality.
        heads (int): Number of attention heads (default: :obj:`8`).
        dim_head (int): Dimensionality of each attention head (default: :obj:`16`).
        attn_dropout (float): Dropout rate for the attention module (default: :obj:`0.3`).
        ff_dropout (float): Dropout rate for the feedforward module (default: :obj:`0.3`).
    """

    def __init__(
        self,
        dim,
        heads: int = 8,
        dim_head: int = 16,
        attn_dropout: float = 0.3,
        ff_dropout: float = 0.3,
    ):
        super().__init__()
        self.attn = PreNorm(
            dim=dim,
            fn=SelfAttention(
                dim=dim,
                heads=heads,
                dim_head=dim_head,
                dropout=attn_dropout,
            ),
        )
        self.ff = PreNorm(
            dim=dim,
            fn=FeedForward(dim=dim, dropout=ff_dropout),
        )

    def reset_parameters(self) -> None:
        self.attn.reset_parameters()
        self.ff.reset_parameters()

    def forward(self, x: Tensor, return_attn: bool = False) -> Tensor:
        attn_out, post_softmax_attn = self.attn(x)
        x = x + attn_out
        x = self.ff(x) + x
        if return_attn:
            return x, post_softmax_attn
        return x


class SAINTConv(torch.nn.Module):
    r"""The SAINTConv Layer introduced in the
    `"SAINT: Improved Neural Networks for Tabular Data via Row Attention
        and Contrastive Pre-Training"
    <https://arxiv.org/abs/2106.01342>`_ paper.

    Args:
        in_dim (int): Input channel dimensionality.
        feat_num (int): Number of features.
        heads (int): Number of attention heads (default: :obj:`8`).
        dim_head (int): Dimensionality of each attention head (default: :obj:`16`).
        attn_dropout (float): Attention module dropout (default: :obj:`0.3`).
        ff_dropout (float): Feedforward module dropout (default: :obj:`0.3`).
        metadata (Dict[ColType, List[Dict[str, Any]]], optional):
            Metadata for the pre-encoder (default: :obj:`None`).
    """

    def __init__(
        self,
        in_dim,
        feat_num,
        heads: int = 8,
        dim_head: int = 16,
        attn_dropout: float = 0.3,
        ff_dropout: float = 0.3,
        metadata: Dict[ColType, List[Dict[str, Any]]] = None,
    ):
        super().__init__()
        # Column Transformer
        self.col_transformer = Transformer(
            dim=in_dim,
            heads=heads,
            dim_head=dim_head,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
        )
        # Row Transformer
        row_dim = in_dim * feat_num
        self.row_transformer = Transformer(
            dim=row_dim,
            heads=heads,
            dim_head=dim_head,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
        )

        self.pre_encoder = None
        if metadata:
            self.pre_encoder = FTTransformerPreEncoder(
                out_dim=in_dim,
                metadata=metadata,
            )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.col_transformer.reset_parameters()
        self.row_transformer.reset_parameters()
        if self.pre_encoder is not None:
            self.pre_encoder.reset_parameters()

    def forward(self, x):
        if self.pre_encoder is not None:
            x = self.pre_encoder(x)
        x = self.col_transformer(x)
        shape = x.shape
        x = x.reshape(1, x.shape[0], -1)
        x = self.row_transformer(x)
        return x.reshape(shape)
