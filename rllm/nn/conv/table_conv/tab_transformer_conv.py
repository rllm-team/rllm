from __future__ import annotations
from typing import Union, Dict, List, Any

import torch
from torch import Tensor

from rllm.types import ColType
from rllm.nn.pre_encoder import TabTransformerPreEncoder


class TabTransformerConv(torch.nn.Module):
    r"""The TabTransformer LayerConv introduced in the
    `"TabTransformer: Tabular Data Modeling Using Contextual Embeddings"
    <https://arxiv.org/abs/2012.06678>`_ paper.

    This layer leverages the power of the Transformer architecture to capture
    complex patterns and relationships within the categorical data.

    Args:
        conv_dim (int): Input/Output dimensionality.
        num_heads (int, optional): Number of attention heads (default: 8).
        dropout (float, optional): Attention module dropout (default: 0.3).
        activation (str, optional): Activation function (default: "relu").
        use_pre_encoder (bool): Whether to use a pre-encoder (default: :obj:`False`).
        metadata (Dict[ColType, List[Dict[str, Any]]], optional):
            Metadata for each column type, specifying the statistics and
            properties of the columns. (default: :obj:`None`).
    """

    def __init__(
        self,
        conv_dim: int,
        num_heads: int = 8,
        dropout: float = 0.3,
        activation: str = "relu",
        use_pre_encoder: bool = False,
        metadata: Dict[ColType, List[Dict[str, Any]]] = None,
    ):
        super().__init__()
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=conv_dim,
            nhead=num_heads,
            dim_feedforward=conv_dim,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )
        encoder_norm = torch.nn.LayerNorm(conv_dim)
        self.transformer = torch.nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=1,
            norm=encoder_norm,
        )

        # Define PreEncoder
        self.pre_encoder = None
        if use_pre_encoder:
            self.pre_encoder = TabTransformerPreEncoder(
                out_dim=conv_dim,
                metadata=metadata,
            )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.pre_encoder is not None:
            self.pre_encoder.reset_parameters()

    def forward(self, x: Union[Dict, Tensor]):
        if self.pre_encoder is not None:
            x = self.pre_encoder(x, return_dict=True)

        x[ColType.CATEGORICAL] = self.transformer(x[ColType.CATEGORICAL])

        return x
