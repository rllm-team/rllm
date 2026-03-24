from __future__ import annotations
from typing import Union, Dict

import torch
from torch import Tensor

from rllm.types import ColType


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

    Example:
        >>> import torch
        >>> from rllm.types import ColType
        >>> conv = TabTransformerConv(conv_dim=32, num_heads=8, dropout=0.1)
        >>> x = {ColType.CATEGORICAL: torch.randn(8, 10, 32)}
        >>> out = conv(x)
    """

    def __init__(
        self,
        conv_dim: int,
        num_heads: int = 8,
        dropout: float = 0.3,
        activation: str = "relu",
    ):
        super().__init__()
        # One encoder layer models contextual interactions among categorical columns.
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

    def forward(self, x: Union[Dict, Tensor]) -> Union[Dict, Tensor]:
        """Encode categorical features with self-attention.

        Args:
            x (Union[Dict, Tensor]): A container that supports
                ``x[ColType.CATEGORICAL]`` indexing. The categorical tensor
                is typically shaped as ``[batch_size, num_categorical_cols, conv_dim]``.

        Returns:
            Union[Dict, Tensor]: The same container type as input, where
            ``x[ColType.CATEGORICAL]`` is replaced with the transformed tensor.
        """
        if isinstance(x, dict) and ColType.CATEGORICAL in x:
            x[ColType.CATEGORICAL] = self.transformer(x[ColType.CATEGORICAL])
        elif isinstance(x, Tensor):
            x = self.transformer(x)
        return x
