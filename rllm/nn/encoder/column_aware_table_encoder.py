from typing import Any, Dict, List, Type

import torch
from torch import Tensor

from rllm.data import TableData
from rllm.nn.conv.table_conv import TabTransformerConv
from rllm.nn.encoder.tab_transformer_pre_encoder import TabTransformerPreEncoder
from rllm.types import ColType, StatType


class ColumnAwareTableEncoder(torch.nn.Module):
    r"""Reusable column-aware encoder for a single table.

    This module is the reusable implementation point for the column-aware
    table encoder in Section 3.1 of the InRTL paper. It receives one
    :class:`~rllm.data.TableData` instance and returns one embedding per row.

    The current default path deliberately preserves the existing InRTL example
    behavior: type-aware pre-encoding, table convolution over column tokens,
    concatenation, and mean pooling. The paper's learned column weighting and
    ResNet fusion in Eq. 3-4 are not enabled here yet, because introducing
    them would change existing model outputs.

    Args:
        in_dim (int): Retained for API compatibility with existing table
            encoders. The input feature dimensions are inferred from metadata.
        out_dim (int): Shared embedding dimension for every column token.
        num_layers (int): Number of table-convolution layers. (default: ``1``)
        metadata (Dict[ColType, List[Dict[StatType, Any]]], optional): Column
            metadata used by the type-aware pre-encoder.
        table_conv (Type[torch.nn.Module]): Table-convolution layer applied to
            the encoded column tokens. (default: :class:`TabTransformerConv`)
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_layers: int = 1,
        metadata: Dict[ColType, List[Dict[StatType, Any]]] = None,
        table_conv: Type[torch.nn.Module] = TabTransformerConv,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.pre_encoder = TabTransformerPreEncoder(out_dim=out_dim, metadata=metadata)
        self.convs = torch.nn.ModuleList(
            [table_conv(conv_dim=out_dim) for _ in range(num_layers)]
        )

    def forward(self, table: TableData) -> Tensor:
        r"""Encode each row of ``table`` into a shared embedding space."""
        x = self.pre_encoder(table.feat_dict, return_dict=True)
        for conv in self.convs:
            x = conv(x)
        return torch.cat(list(x.values()), dim=1).mean(dim=1)
