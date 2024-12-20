from __future__ import annotations
from typing import Dict, List, Any

import torch
from torch.nn import (
    LayerNorm,
    TransformerEncoderLayer,
    TransformerEncoder,
)

from rllm.types import ColType
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
        num_heads: int = 8,
        dropout: float = 0.3,
        activation: str = "relu",
        metadata: Dict[ColType, List[Dict[str, Any]]] = None,
    ):
        super().__init__()

        # Column Transformer
        col_encoder_layer = TransformerEncoderLayer(
            d_model=in_dim,
            nhead=num_heads,
            dim_feedforward=in_dim,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )
        col_encoder_norm = LayerNorm(in_dim)
        self.col_transformer = TransformerEncoder(
            encoder_layer=col_encoder_layer,
            num_layers=1,
            norm=col_encoder_norm,
        )

        # Row Transformer
        row_encoder_layer = TransformerEncoderLayer(
            d_model=in_dim * feat_num,
            nhead=num_heads,
            dim_feedforward=in_dim * feat_num,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )
        row_encoder_norm = LayerNorm(in_dim * feat_num)
        self.row_transformer = TransformerEncoder(
            encoder_layer=row_encoder_layer,
            num_layers=1,
            norm=row_encoder_norm,
        )

        # Define PreEncoder
        self.pre_encoder = None
        if metadata:
            self.pre_encoder = FTTransformerPreEncoder(
                out_dim=in_dim,
                metadata=metadata,
            )

        self.reset_parameters()

    def reset_parameters(self) -> None:
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
