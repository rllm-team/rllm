from __future__ import annotations
from typing import Union, Dict, List, Any

import torch
from torch import Tensor

from rllm.types import ColType
from rllm.nn.pre_encoder import FTTransformerPreEncoder


class SAINTConv(torch.nn.Module):
    r"""The SAINTConv Layer introduced in the
    `"SAINT: Improved Neural Networks for Tabular Data via Row Attention
    and Contrastive Pre-Training" <https://arxiv.org/abs/2106.01342>`__ paper.

    This layer applies two :obj:`TransformerEncoder` modules: one for aggregating
    information between columns, and another for aggregating information
    between samples. This dual attention mechanism allows the model to capture
    complex relationships both within the features of a single sample and
    across different samples.

    Args:
        conv_dim (int): Input/Output dimensionality.
        num_feats (int): Number of features.
        num_heads (int, optional): Number of attention heads (default: 8).
        dropout (float, optional): Attention module dropout (default: 0.3).
        activation (str, optional): Activation function (default: "relu").
        use_pre_encoder (bool, optional): Whether to use a pre-encoder (default: :obj:`False`).
        metadata (Dict[rllm.types.ColType, List[Dict[str, Any]]], optional):
            Metadata for each column type, specifying the statistics and
            properties of the columns. (default: :obj:`None`).
    """

    def __init__(
        self,
        conv_dim: int,
        num_feats: int,
        num_heads: int = 8,
        dropout: float = 0.3,
        activation: str = "relu",
        use_pre_encoder: bool = False,
        metadata: Dict[ColType, List[Dict[str, Any]]] = None,
    ):
        super().__init__()

        # Column Transformer
        col_encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=conv_dim,
            nhead=num_heads,
            dim_feedforward=conv_dim,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )
        col_encoder_norm = torch.nn.LayerNorm(conv_dim)
        self.col_transformer = torch.nn.TransformerEncoder(
            encoder_layer=col_encoder_layer,
            num_layers=1,
            norm=col_encoder_norm,
        )

        # Row Transformer
        row_encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=conv_dim * num_feats,
            nhead=num_heads,
            dim_feedforward=conv_dim * num_feats,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )
        row_encoder_norm = torch.nn.LayerNorm(conv_dim * num_feats)
        self.row_transformer = torch.nn.TransformerEncoder(
            encoder_layer=row_encoder_layer,
            num_layers=1,
            norm=row_encoder_norm,
        )

        # Define PreEncoder
        self.pre_encoder = None
        if use_pre_encoder:
            self.pre_encoder = FTTransformerPreEncoder(
                out_dim=conv_dim,
                metadata=metadata,
            )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.pre_encoder is not None:
            self.pre_encoder.reset_parameters()

    def forward(self, x: Union[Dict, Tensor]):
        if self.pre_encoder is not None:
            x = self.pre_encoder(x)
        x = self.col_transformer(x)
        shape = x.shape
        x = x.reshape(1, x.shape[0], -1)
        x = self.row_transformer(x)
        return x.reshape(shape)
