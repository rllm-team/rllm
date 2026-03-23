from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import (
    Linear,
    LayerNorm,
    BatchNorm1d,
    Dropout,
    ReLU,
)


class ResNetConv(torch.nn.Module):
    r"""The ResNet-like TNN LayerConv introduced in the
    `"Revisiting Deep Learning Models for Tabular Data"
    <https://arxiv.org/abs/2106.11959>`_ paper.

    This module applies a two-layer MLP block with optional normalization,
    activation, and dropout, then adds a residual shortcut connection.

    Args:
        in_dim (int): Input feature dimensionality.
        out_dim (int): Output feature dimensionality.
        normalization (str | None): Normalization type. Supported values are
            :obj:`"layer_norm"`, :obj:`"batch_norm"`, or :obj:`None`.
            (default: :obj:`"layer_norm"`)
        dropout (float): Dropout probability. (default: :obj:`0.0`)

    Example:
        >>> import torch
        >>> conv = ResNetConv(in_dim=16, out_dim=32, normalization="layer_norm", dropout=0.1)
        >>> x = torch.randn(64, 16)
        >>> out = conv(x)
        >>> out.shape
        torch.Size([64, 32])
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        normalization: str | None = "layer_norm",
        dropout: float = 0.0,
    ):
        super().__init__()

        self.lin1 = Linear(in_dim, out_dim)
        self.lin2 = Linear(out_dim, out_dim)

        self.norm1 = None
        self.norm2 = None

        if normalization == "layer_norm":
            self.norm1 = LayerNorm(out_dim)
            self.norm2 = LayerNorm(out_dim)
        elif normalization == "batch_norm":
            self.norm1 = BatchNorm1d(out_dim)
            self.norm2 = BatchNorm1d(out_dim)
        else:
            self.norm1 = None
            self.norm2 = None

        if in_dim != out_dim:
            self.short_cut = Linear(in_dim, out_dim)
        else:
            self.short_cut = None

        self.relu = ReLU()
        self.dropout = Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        r"""Resets all learnable parameters of the module."""
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        if self.norm1 is not None:
            self.norm1.reset_parameters()
        if self.norm2 is not None:
            self.norm2.reset_parameters()
        if self.short_cut is not None:
            self.short_cut.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        r"""Apply residual MLP transformation.

        Args:
            x (Tensor): Input tensor of shape :obj:`[..., in_dim]`.

        Returns:
            Tensor: Output tensor of shape :obj:`[..., out_dim]`.
        """
        residual = x

        x = self.lin1(x)
        x = self.norm1(x) if self.norm1 is not None else x
        x = self.relu(x)
        x = self.dropout(x)

        x = self.lin2(x)
        x = self.norm2(x) if self.norm2 is not None else x
        x = self.relu(x)
        x = self.dropout(x)

        if self.short_cut is not None:
            residual = self.short_cut(residual)

        x = x + residual
        return x
