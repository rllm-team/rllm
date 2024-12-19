from __future__ import annotations
from typing import Dict, List, Any

import torch

from rllm.types import ColType
from rllm.nn.conv.utils import Transformer
from rllm.nn.pre_encoder import FTTransformerPreEncoder


class SAINTConv(torch.nn.Module):
    r"""The SAINTConv Layer introduced in the
    `"SAINT: Improved Neural Networks for Tabular Data via Row Attention
        and Contrastive Pre-Training"
    <https://arxiv.org/abs/2106.01342>`_ paper.

    Args:
        in_dim (int): Input channel dimensionality.
        feat_num (int): Number of features.
        heads (int): Number of attention heads (default: :obj:`8`).
        head_dim (int): Dimensionality of each attention head (default: :obj:`16`).
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
        head_dim: int = 16,
        attn_dropout: float = 0.3,
        ff_dropout: float = 0.3,
        metadata: Dict[ColType, List[Dict[str, Any]]] = None,
    ):
        super().__init__()
        # Column Transformer
        self.col_transformer = Transformer(
            dim=in_dim,
            heads=heads,
            head_dim=head_dim,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
        )
        # Row Transformer
        row_dim = in_dim * feat_num
        self.row_transformer = Transformer(
            dim=row_dim,
            heads=heads,
            head_dim=head_dim,
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
