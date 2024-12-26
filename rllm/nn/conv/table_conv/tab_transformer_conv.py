from __future__ import annotations
from typing import Dict, List, Any

import torch
from torch.nn import (
    LayerNorm,
    TransformerEncoder,
    TransformerEncoderLayer,
)

from rllm.types import ColType
from rllm.nn.pre_encoder import TabTransformerPreEncoder


class TabTransformerConv(torch.nn.Module):
    r"""The TabTransformer LayerConv introduced in the
    `"TabTransformer: Tabular Data Modeling Using Contextual Embeddings"
    <https://arxiv.org/abs/2012.06678>`_ paper.

    Args:
        dim (int): The input/output channel dimensionality.
        num_heads (int, optional): Number of attention heads (default: 8).
        dropout (float, optional): Dropout rate for the attention module (default: 0.3).
        activation (str, optional): Activation function to use (default: "gelu").
        metadata (Dict[ColType, List[Dict[str, Any]]], optional): Metadata for the pre-encoder (default: None).
    """

    def __init__(
        self,
        dim,
        num_heads: int = 8,
        dropout: float = 0.3,
        activation: str = "relu",
        metadata: Dict[ColType, List[Dict[str, Any]]] = None,
    ):
        super().__init__()
        encoder_layer = TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )
        encoder_norm = LayerNorm(dim)
        self.transformer = TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=1,
            norm=encoder_norm,
        )

        self.pre_encoder = None
        if metadata:
            self.pre_encoder = TabTransformerPreEncoder(
                out_dim=dim,
                metadata=metadata,
            )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.pre_encoder is not None:
            self.pre_encoder.reset_parameters()

    def forward(self, x):
        if self.pre_encoder is not None:
            x = self.pre_encoder(x, return_dict=True)

        x[ColType.CATEGORICAL] = self.transformer(x[ColType.CATEGORICAL])

        return x
