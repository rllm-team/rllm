from typing import Any, Dict, List, Optional, Type, Union

import torch
from torch import Tensor

from rllm.data import TableData
from rllm.types import ColType
from rllm.nn.conv.table_conv import TabTransformerConv
from .base_encoder import BaseEncoder
from .tab_transformer_pre_encoder import TabTransformerPreEncoder


class TableEncoder(BaseEncoder):
    r"""TableEncoder is a table-level encoder used by BRIDGE.

    It first applies a pre-encoder to each column type, then stacks table
    convolution layers and finally pools across columns.

    Args:
        in_dim (int): Input dimensionality of the table data.
        out_dim (int): Output dimensionality for the encoded table data.
        num_layers (int, optional): Number of convolution layers.
            (default: :obj:`1`).
        metadata (Dict[ColType, List[Dict[str, Any]]], optional):
            Metadata for each column type.
        table_conv (Type[torch.nn.Module], optional):
            The convolution module used for table encoding.
            (default: :obj:`rllm.nn.conv.table_conv.TabTransformerConv`).
        pre_encoder (Type[torch.nn.Module], optional):
            The pre-encoder used before table convolution.
            (default: :obj:`rllm.nn.encoder.TabTransformerPreEncoder`).
        pre_encoder_return_dict (bool, optional):
            Whether to ask pre-encoder to return a feature dict.
            (default: :obj:`True`).
        pooling (str, optional):
            Output pooling strategy after stacked table conv layers.
            Supported values: :obj:`"mean"`, :obj:`"first"`,
            :obj:`"flatten"`, :obj:`"none"`.
            If :obj:`"none"`, no pooling is applied.
            (default: :obj:`"mean"`).
        pre_encoder_kwargs (Dict[str, Any], optional):
            Extra keyword args passed to pre-encoder constructor.
        table_conv_kwargs (Dict[str, Any], optional):
            Extra keyword args passed to table convolution constructor.

    Returns:
        This class does not return tensors in ``__init__``.
        The ``forward`` method returns pooled table embeddings.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_layers: int = 1,
        metadata: Dict[ColType, List[Dict[str, Any]]] = None,
        table_conv: Type[torch.nn.Module] = TabTransformerConv,
        pre_encoder: Type[torch.nn.Module] = TabTransformerPreEncoder,
        pre_encoder_return_dict: bool = True,
        pooling: str = "mean",
        pre_encoder_kwargs: Optional[Dict[str, Any]] = None,
        table_conv_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(num_layers=num_layers)

        self.pre_encoder_return_dict = pre_encoder_return_dict
        self.pooling = pooling

        pre_encoder_kwargs = pre_encoder_kwargs or {}
        table_conv_kwargs = table_conv_kwargs or {}

        self.encoder = pre_encoder(
            out_dim=out_dim,
            metadata=metadata,
            **pre_encoder_kwargs,
        )
        for _ in range(num_layers):
            self.convs.append(table_conv(conv_dim=out_dim, **table_conv_kwargs))

    def forward(self, table: TableData) -> Union[Tensor, Dict[ColType, Tensor]]:
        """Encode a table into a pooled feature representation.

        Args:
            table (TableData): Input table data object.

        Returns:
            Union[Tensor, Dict[ColType, Tensor]]: Encoded output after table
            backbone. If pooling is :obj:`"none"`, this returns the raw
            convolution output without pooling.
        """
        x = table.feat_dict
        x = self.encoder(x, return_dict=self.pre_encoder_return_dict)
        for conv in self.convs:
            x = conv(x)
        return x
