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
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        if self.norm1 is not None:
            self.norm1.reset_parameters()
        if self.norm2 is not None:
            self.norm2.reset_parameters()
        if self.short_cut is not None:
            self.short_cut.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
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
