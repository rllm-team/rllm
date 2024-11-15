from __future__ import annotations

import torch
from typing import Tuple
from torch import Tensor


class ExcelFormerConv(torch.nn.Module):
    r"""The ExcelFormerConv Layer introduced in the
    `"ExcelFormer: A neural network surpassing GBDTs on tabular data"
    <https://arxiv.org/abs/2301.02819>`_ paper.

    Args:
        dim (int): Input/output channel dimensionality.
        heads (int): Number of attention heads.
        dim_head (int):  Dimensionality of each attention head.
        dropout (float): attention module dropout (default: :obj:`0.3`).
    """

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 16,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.layer_norm = torch.nn.LayerNorm(dim)
        self.sp_attention = SemiPermeableAttention(
            dim=dim, heads=heads, dim_head=dim_head, dropout=dropout
        )
        self.GLU_layer = GLU_layer(dim, dim)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.layer_norm(x)
        x = self.sp_attention(x)
        x = x + self.GLU_layer(x)
        return x


class SemiPermeableAttention(torch.nn.Module):
    r"""Semi-Permeable Attention module propose in the
    `"ExcelFormer: A neural network surpassing GBDTs on tabular data"`
    <https://arxiv.org/abs/2301.02819>`_ paper.

    Args:
        dim (int): Input channel dimensionality
        heads (int): Number of heads in Attention module (default: :obj:`8.`)
        dim_head(int): Dimension of each attention head (default: :obj:`16.`)
        dropout (float): Percentage of random deactivation (default: :obj:`0.`)
    """

    def __init__(self, dim, heads=8, dim_head=16, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5

        self.to_qkv = torch.nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = torch.nn.Linear(inner_dim, dim)

        self.dropout = torch.nn.Dropout(dropout)

    def _rearrange_qkv(self, x: Tensor) -> Tensor:
        # reshape b n (h d) -> b h n d
        b, num_cols, dim = x.shape
        d_head = dim // self.heads
        x = x.reshape(b, num_cols, self.heads, d_head)
        x = x.permute(0, 2, 1, 3)
        return x

    def forward(self, x: Tensor):
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q = self._rearrange_qkv(q)
        k = self._rearrange_qkv(k)
        v = self._rearrange_qkv(v)
        sim = torch.einsum("b h i d, b h j d -> b h i j", q, k)
        mask = self.get_attention_mask(sim.size(), sim.device)
        attn = (sim + mask) * self.scale
        attn = attn.softmax(dim=-1)
        dropped_attn = self.dropout(attn)

        out = torch.einsum("b h i j, b h j d -> b h i d", dropped_attn, v)
        # reshape b h n d -> b n (h d)
        out = out.permute(0, 2, 1, 3)
        out = out.reshape(out.size(0), out.size(1), -1)
        return self.to_out(out)

    def reset_parameters(self) -> None:
        self.to_qkv.reset_parameters()
        self.to_out.reset_parameters()

    def get_attention_mask(self, input_shape, device):
        bs, heads, seq_len, _ = input_shape
        seq_ids = torch.arange(seq_len, device=device)
        attention_mask = (
            seq_ids[None, None, :].repeat(bs, seq_len, 1) <= seq_ids[None, :, None]
        )
        attention_mask = (1.0 - attention_mask.float()) * -1e4
        attention_mask = attention_mask.unsqueeze(1).repeat(1, heads, 1, 1)
        return attention_mask


class GLU_layer(torch.nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        fc=None,
    ):
        super().__init__()

        self.out_dim = out_dim
        if fc:
            self.fc = fc
        else:
            self.fc = torch.nn.Linear(in_dim, 2 * out_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc(x)
        x, gates = x.chunk(2, dim=2)
        return x * torch.nn.functional.tanh(gates)
