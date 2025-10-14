from __future__ import annotations
from typing import Optional, Union, Dict, List, Any

import torch
from torch import Tensor
from torch.nn import Parameter

from rllm.types import ColType
from rllm.nn.pre_encoder import FTTransformerPreEncoder


class FTTransformerConv(torch.nn.Module):
    r"""The FT-Transformer backbone in the `"Revisiting Deep Learning
    Models for Tabular Data" <https://arxiv.org/abs/2106.11959>`_ paper.

    This module concatenates a learnable CLS token embedding :obj:`x_cls` to
    the input tensor :obj:`x` and applies a multi-layer Transformer on the
    concatenated tensor. After the Transformer layer, the output tensor is
    divided into two parts: (1) :obj:`x`, corresponding to the original input
    tensor, and (2) :obj:`x_cls`, corresponding to the CLS token tensor.

    Args:
        conv_dim (int): Input/Output dimensionality.
        feedforward_dim (int, optional): Hidden dimensionality used by
            feedforward network of the Transformer model. If :obj:`None`, it
            will be set to :obj:`dim` (default: :obj:`None`)
        layers (int): Number of transformer encoder layers. (default: 3)
        num_heads (int): Number of heads in multi-head attention (default: 8)
        dropout (int): The dropout value (default: 0.1)
        activation (str): The activation function (default: :obj:`relu`)
        use_cls (bool): Whether to use a CLS token (default: :obj:`False`).
        use_pre_encoder (bool): Whether to use a pre-encoder (default: :obj:`False`).
        metadata (Optional[Dict[ColType, List[Dict[str, Any]]]]): Metadata for
            each column type, specifying the statistics and properties of the
            columns (default: :obj:`None`).
    """

    def __init__(
        self,
        conv_dim: int,
        feedforward_dim: Optional[int] = None,
        num_heads: int = 8,
        dropout: float = 0.3,
        activation: str = "relu",
        use_cls: bool = False,
        use_pre_encoder: bool = False,
        metadata: Dict[ColType, List[Dict[str, Any]]] = None,
    ):
        super().__init__()
        self.use_cls = use_cls
        self.metadata = metadata
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=conv_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim or conv_dim,
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
        self.cls_embedding = Parameter(torch.empty(conv_dim))

        # Define PreEncoder
        self.pre_encoder = None
        if use_pre_encoder:
            self.pre_encoder = FTTransformerPreEncoder(
                out_dim=conv_dim,
                metadata=self.metadata,
            )

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.normal_(self.cls_embedding, std=0.01)
        for p in self.transformer.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)
        if self.pre_encoder:
            self.pre_encoder.reset_parameters()

    def forward(self, x: Union[Dict, Tensor]) -> Tensor:
        r"""CLS-token augmented Transformer convolution.

        Args:
            x (Union[Dict, Tensor]): Input tensor of shape [batch_size, num_cols, dim]

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, num_cols, dim]
            corresponding to the input columns, or output tensor of shape
            [batch_size, num_cols + 1, dim], corresponding to the
            added CLS token column.
        """
        if self.pre_encoder is not None:
            x = self.pre_encoder(x)

        B, _, _ = x.shape
        # [batch_size, num_cols, dim]
        x_cls = self.cls_embedding.repeat(B, 1, 1)
        # [batch_size, num_cols + 1, dim]
        x_concat = torch.cat([x_cls, x], dim=1)
        # [batch_size, num_cols + 1, dim]
        x_concat = self.transformer(x_concat)
        if self.use_cls:
            return x_concat[:, 0, :]
        return x_concat[:, 1:, :]
