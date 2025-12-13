from __future__ import annotations
import math
from typing import Dict, List, Any

import torch
from torch import Tensor
from torch.nn import Sequential

from rllm.types import ColType
from rllm.data import TableData
from rllm.nn.pre_encoder import ResNetPreEncoder
from rllm.nn.conv.table_conv import ResNetConv


class TableResNet(torch.nn.Module):
    r"""The ResNet-like TNN introduced in the
    `"Revisiting Deep Learning Models for Tabular Data"
    <https://arxiv.org/abs/2106.11959>`_ paper.

    Args:
        hidden_dim (int): The hidden dimension.
        out_dim (int): The output dimension.
        num_layers (int): The number of layers.
        metadata (Dict[ColType, List[Dict[str, Any]]]): The metadata of the table.
        normalization (str | None): The normalization method.
        dropout (float): The dropout rate.
    """
    def __init__(
        self,
        hidden_dim: int,
        out_dim: int,
        num_layers: int,
        metadata: Dict[ColType, List[Dict[str, Any]]],
        normalization: str | None = "layer_norm",
        dropout: float = 0.2,
    ):
        super().__init__()
        self.normalization = normalization
        self.dropout = dropout

        # PreEncoder
        self.pre_encoder = ResNetPreEncoder(
            out_dim=hidden_dim,
            metadata=metadata,
        )

        # ConvLayers
        n_cols = [
            len(metadata[coltype])
            for coltype in metadata.keys()
        ]
        n_cols = sum(n_cols)
        conv_in_dim = hidden_dim * n_cols
        self.convs = Sequential(*[
            ResNetConv(
                in_dim = conv_in_dim if i == 0 else hidden_dim,
                out_dim = hidden_dim,
                normalization=normalization,
                dropout=dropout,
            ) for i in range(num_layers)
        ])

        # Decoder
        self.decoder = Sequential(
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, out_dim),
        )
        self.reset_parameters()

    def reset_parameters(self):
        self.pre_encoder.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for layer in self.decoder:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(self, table: TableData) -> Tensor:
        x = table.feat_dict

        x = self.pre_encoder(x) # (B, n_cols, hidden_dim)
        # flatten the pre_encoder output
        x = x.view(x.size(0), math.prod(x.shape[1:]))   # (B, n_cols * hidden_dim)
        x = self.convs(x) # (B, hidden_dim)
        x = self.decoder(x) # (B, hidden_dim)
        return x