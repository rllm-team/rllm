from __future__ import annotations

import torch
from torch import Tensor


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
        num_cols (int): Number of features.
        num_heads (int, optional): Number of attention heads (default: 8).
        dropout (float, optional): Attention module dropout (default: 0.3).
        activation (str, optional): Activation function (default: "relu").

    Example:
        >>> import torch
        >>> conv = SAINTConv(conv_dim=16, num_cols=8, num_heads=4, dropout=0.1)
        >>> x = torch.randn(32, 8, 16)
        >>> out = conv(x)
    """

    def __init__(
        self,
        conv_dim: int,
        num_cols: int,
        num_heads: int = 8,
        dropout: float = 0.3,
        activation: str = "relu",
    ):
        super().__init__()

        # Column Transformer
        col_encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=conv_dim,
            nhead=num_heads,
            dim_feedforward=conv_dim * 4,
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
            d_model=conv_dim * num_cols,
            nhead=num_heads,
            dim_feedforward=conv_dim * num_cols * 4,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )
        row_encoder_norm = torch.nn.LayerNorm(conv_dim * num_cols)
        self.row_transformer = torch.nn.TransformerEncoder(
            encoder_layer=row_encoder_layer,
            num_layers=1,
            norm=row_encoder_norm,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply column attention then row attention.

        Args:
            x (Tensor): Input tensor of shape
                ``[batch_size, num_cols, conv_dim]``.

        Returns:
            torch.Tensor: Output tensor with the same shape as input.
        """
        x = self.col_transformer(x)
        shape = x.shape
        # Flatten feature dimension for row-wise attention across samples.
        x = x.reshape(1, x.shape[0], -1)
        x = self.row_transformer(x)
        return x.reshape(shape)
