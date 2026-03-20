from __future__ import annotations
from typing import Optional

import torch
from torch import Tensor
from torch.nn import Parameter


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
            will be set to :obj:`conv_dim` (default: :obj:`None`).
        num_heads (int): Number of heads in multi-head attention (default: 8)
        dropout (float): The dropout value (default: 0.3)
        activation (str): The activation function (default: :obj:`relu`)
        use_cls (bool): Whether to use a CLS token (default: :obj:`False`).

    Returns:
        This class does not return a tensor in ``__init__``.
        The ``forward`` method returns either column embeddings or the CLS
        embedding, depending on ``use_cls``.

    Example:
        >>> import torch
        >>> conv = FTTransformerConv(conv_dim=32, num_heads=8, use_cls=False)
        >>> x = torch.randn(16, 10, 32)
        >>> out = conv(x)
        >>> out.shape
        torch.Size([16, 10, 32])
    """

    def __init__(
        self,
        conv_dim: int,
        feedforward_dim: Optional[int] = None,
        num_heads: int = 8,
        dropout: float = 0.3,
        activation: str = "relu",
        use_cls: bool = False,
    ):
        super().__init__()
        self.use_cls = use_cls
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

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.normal_(self.cls_embedding, std=0.01)
        for p in self.transformer.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def forward(self, x: Tensor) -> Tensor:
        r"""CLS-token augmented Transformer convolution.

        Args:
            x (Tensor): Input tensor of shape [batch_size, num_cols, dim]

        Returns:
            torch.Tensor: If ``use_cls=False``, output tensor of shape
            ``[batch_size, num_cols, dim]`` corresponding to input columns.
            If ``use_cls=True``, output tensor of shape
            ``[batch_size, dim]`` for the CLS token representation.
        """

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
