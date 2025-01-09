from __future__ import annotations
from typing import Tuple, Union, Dict, List, Any

import torch
from torch import Tensor

from rllm.types import ColType
from rllm.nn.pre_encoder import FTTransformerPreEncoder


class GLULayer(torch.nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
    ):
        super().__init__()
        self.fc = torch.nn.Linear(in_dim, 2 * out_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.fc.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc(x)
        x, gates = x.chunk(2, dim=2)
        return x * torch.nn.functional.tanh(gates)


class SemiPermeableAttention(torch.nn.Module):
    r"""Semi-Permeable Attention module propose in the
    `"ExcelFormer: A neural network surpassing GBDTs on tabular data"`
    <https://arxiv.org/abs/2301.02819>`_ paper.

    Args:
        dim (int): Input dimensionality
        num_heads (int): Number of heads in Attention module (default: :obj:`8`)
        head_dim(int): Dimension of each attention head (default: :obj:`16`)
        dropout (float): Percentage of random deactivation (default: :obj:`0.`)
    """

    def __init__(self, dim, num_heads=8, head_dim=16, dropout=0.0):
        super().__init__()
        inner_dim = head_dim * num_heads
        self.num_heads = num_heads
        self.scale = head_dim**-0.5

        self.to_qkv = torch.nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = torch.nn.Linear(inner_dim, dim)

        self.dropout = torch.nn.Dropout(dropout)

    def _rearrange_qkv(self, x: Tensor) -> Tensor:
        # reshape b n (h d) -> b h n d
        b, num_cols, dim = x.shape
        d_head = dim // self.num_heads
        x = x.reshape(b, num_cols, self.num_heads, d_head)
        x = x.permute(0, 2, 1, 3)
        return x

    def forward(self, x: Tensor):
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q = self._rearrange_qkv(q)
        k = self._rearrange_qkv(k)
        v = self._rearrange_qkv(v)
        sim = torch.einsum("b h i d, b h j d -> b h i j", q, k)
        mask = self.get_attention_mask(input_shape=sim.size(), device=sim.device)
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

    def get_attention_mask(self, input_shape: Tuple, device):
        bs, num_heads, seq_len, _ = input_shape
        seq_ids = torch.arange(seq_len, device=device)
        attention_mask = (
            seq_ids[None, None, :].repeat(bs, seq_len, 1) <= seq_ids[None, :, None]
        )
        attention_mask = (1.0 - attention_mask.float()) * -1e4
        attention_mask = attention_mask.unsqueeze(1).repeat(1, num_heads, 1, 1)
        return attention_mask


class ExcelFormerConv(torch.nn.Module):
    r"""The ExcelFormerConv Layer introduced in the
    `"ExcelFormer: A neural network surpassing GBDTs on tabular data"
    <https://arxiv.org/abs/2301.02819>`_ paper.

    This layer is designed to handle tabular data by applying a combination of
    normalization, attention, and gated linear unit (GLU). In essence, it is
    a variant of the attention mechanism tailored for tabular data.  If
    metadata is provided, the pre-encoder is used to preprocess the input data
    before applying the subsequent encoders. The layer normalizes the input,
    applies semi-permeable attention, and then uses a GLU layer to enhance the
    representation learning capability.

    Args:
        conv_dim (int): Input/Output dimensionality.
        num_heads (int): Number of attention heads (default: :obj:`8`).
        head_dim (int):  Dimensionality of each attention head (default: :obj:`16`).
        dropout (float): Attention module dropout (default: :obj:`0.3`).
        use_pre_encoder (bool): Whether to use a pre-encoder (default: :obj:`False`).
        metadata (Dict[rllm.types.ColType, List[Dict[str, Any]]], optional):
            Metadata for each column type, specifying the statistics and
            properties of the columns. (default: :obj:`None`).
    """

    def __init__(
        self,
        conv_dim: int,
        num_heads: int = 8,
        head_dim: int = 16,
        dropout: float = 0.5,
        use_pre_encoder: bool = False,
        metadata: Dict[ColType, List[Dict[str, Any]]] = None,
    ):
        super().__init__()
        self.layer_norm = torch.nn.LayerNorm(conv_dim)
        self.sp_attention = SemiPermeableAttention(
            dim=conv_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout
        )
        self.glu_layer = GLULayer(in_dim=conv_dim, out_dim=conv_dim)

        # Define PreEncoder
        self.pre_encoder = None
        if use_pre_encoder:
            self.pre_encoder = FTTransformerPreEncoder(
                out_dim=conv_dim,
                metadata=metadata,
            )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.layer_norm.reset_parameters()
        self.sp_attention.reset_parameters()
        self.glu_layer.reset_parameters()
        if self.pre_encoder:
            self.pre_encoder.reset_parameters()

    def forward(self, x: Union[Dict, Tensor]) -> Tensor:
        if self.pre_encoder:
            x = self.pre_encoder(x)
        x = self.layer_norm(x)
        x = self.sp_attention(x)
        x = x + self.glu_layer(x)
        return x
