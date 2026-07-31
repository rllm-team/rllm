from __future__ import annotations

import math
from typing import Any, Dict, List

import torch
from torch import Tensor
from torch.nn import LayerNorm, Linear, ModuleList, ReLU, Sequential

from rllm.data import TableData
from rllm.nn.conv.table_conv import ResNetConv
from rllm.nn.encoder.col_encoder._embedding_encoder import EmbeddingEncoder
from rllm.nn.encoder.col_encoder._linear_encoder import LinearEncoder
from rllm.nn.encoder.table_pre_encoder import TablePreEncoder
from rllm.types import ColType, StatType


class ColATEPreEncoder(TablePreEncoder):
    r"""Type-aware column-token encoder used internally by ColATE."""

    def __init__(
        self,
        out_dim: int,
        metadata: Dict[ColType, List[Dict[StatType, Any]]],
    ) -> None:
        col_encoder_dict = {
            ColType.CATEGORICAL: EmbeddingEncoder(),
            ColType.NUMERICAL: LinearEncoder(),
        }
        super().__init__(out_dim, metadata, col_encoder_dict)


class ColATE(torch.nn.Module):
    r"""Column-aware table encoder from the InRTL paper.

    ColATE embeds categorical and numerical columns into a shared token space.
    It then follows Eq. 3-4: column tokens are reweighted with a softmax over
    batch-level mean activations, flattened, and fused by residual MLP blocks.
    It returns one embedding per table row.

    Args:
        channels (int): Shared dimension of column tokens and residual blocks.
        out_channels (int): Output embedding dimension for each table row.
        num_layers (int): Number of residual MLP blocks. (default: ``1``)
        metadata (Dict[ColType, List[Dict[StatType, Any]]]): Column metadata
            used by the type-aware pre-encoder.
        normalization (str | None): Normalization used by residual MLP blocks.
            (default: ``"layer_norm"``)
        dropout (float): Dropout probability in residual MLP blocks.
            (default: ``0.2``)
    """

    def __init__(
        self,
        channels: int,
        out_channels: int,
        num_layers: int = 1,
        metadata: Dict[ColType, List[Dict[StatType, Any]]] | None = None,
        normalization: str | None = "layer_norm",
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if metadata is None:
            raise ValueError("ColATE requires table metadata.")
        if num_layers < 0:
            raise ValueError("num_layers must be non-negative.")

        num_cols = sum(len(col_metadata) for col_metadata in metadata.values())
        if num_cols == 0:
            raise ValueError("ColATE requires at least one input column.")

        self.pre_encoder = ColATEPreEncoder(out_dim=channels, metadata=metadata)
        in_channels = channels * num_cols
        self.backbone = ModuleList(
            [
                ResNetConv(
                    in_dim=in_channels if i == 0 else channels,
                    out_dim=channels,
                    normalization=normalization,
                    dropout=dropout,
                )
                for i in range(num_layers)
            ]
        )
        decoder_in_channels = channels if num_layers > 0 else in_channels
        self.decoder = Sequential(
            LayerNorm(decoder_in_channels),
            ReLU(),
            Linear(decoder_in_channels, out_channels),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.pre_encoder.reset_parameters()
        for block in self.backbone:
            block.reset_parameters()
        for layer in self.decoder:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

    def forward(self, table: TableData) -> Tensor:
        r"""Encode each row of ``table`` into a column-aware embedding."""
        x = self.pre_encoder(table.feat_dict)

        # Eq. 3: derive one set of global column weights for this batch.
        column_scores = x.mean(dim=0, keepdim=True).mean(dim=2, keepdim=True)
        x = torch.softmax(column_scores, dim=1) * x

        # Eq. 4: flatten once, then apply the residual MLP stack.
        x = x.view(x.size(0), math.prod(x.shape[1:]))
        for block in self.backbone:
            x = block(x)
        return self.decoder(x)
