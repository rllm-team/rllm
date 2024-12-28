from __future__ import annotations
from typing import Dict, List, Any

import torch

from rllm.types import ColType
from rllm.nn.pre_encoder import FTTransformerPreEncoder


class SAINTConv(torch.nn.Module):
    r"""The SAINTConv Layer introduced in the
    `"SAINT: Improved Neural Networks for Tabular Data via Row Attention
        and Contrastive Pre-Training"
    <https://arxiv.org/abs/2106.01342>`_ paper.

    Args:
        in_dim (int): Input channel dimensionality.
        num_feats (int): Number of features.
        num_heads (int, optional): Number of attention heads (default: 8).
        dropout (float, optional): Attention module dropout (default: 0.3).
        activation (str, optional): Activation function (default: "relu").
        metadata (Dict[ColType, List[Dict[str, Any]]], optional):
            Metadata for each column type, specifying the statistics and
            properties of the columns. (default: :obj:`None`).
    """

    def __init__(
        self,
        in_dim,
        num_feats,
        num_heads: int = 8,
        dropout: float = 0.3,
        activation: str = "relu",
        metadata: Dict[ColType, List[Dict[str, Any]]] = None,
    ):
        super().__init__()

        # Column Transformer
        col_encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=in_dim,
            nhead=num_heads,
            dim_feedforward=in_dim,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )
        col_encoder_norm = torch.nn.LayerNorm(in_dim)
        self.col_transformer = torch.nn.TransformerEncoder(
            encoder_layer=col_encoder_layer,
            num_layers=1,
            norm=col_encoder_norm,
        )

        # Row Transformer
        row_encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=in_dim * num_feats,
            nhead=num_heads,
            dim_feedforward=in_dim * num_feats,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )
        row_encoder_norm = torch.nn.LayerNorm(in_dim * num_feats)
        self.row_transformer = torch.nn.TransformerEncoder(
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
