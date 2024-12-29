from __future__ import annotations
from typing import Any, Dict, List

import torch
from torch import Tensor
from torch.nn import Parameter

from rllm.types import ColType, StatType
from ._col_encoder import ColEncoder


class LinearEncoder(ColEncoder):
    r"""A linear function based ColEncoder for numerical features. It applies
    a linear transformation using :obj:`torch.einsum` on each raw numerical
    feature and concatenates the output embeddings. Note that the
    implementation does this for all numerical features in a batched manner.

    Args:
        in_dim (int, optional): The input dimensionality
            for each numerical feature (default: :obj:`1`).
        out_dim (int, optional): The output dimensionality (default: :obj:`1`).
        stats_list (List[Dict[rllm.types.StatType, Any]], optional):
            The list of statistics for each column within the same column type
            (default: :obj:`None`).
        post_module (torch.nn.Module, optional): The post-hoc module applied
            to the output, such as activation function and normalization. Must
            preserve the shape of the output. If :obj:`None`, no module will
            be applied to the output (default: :obj:`None`).
        activation (torch.nn.Module, optional): The activation function
            applied after the linear transformation. If :obj:`None`,
            no activation function will be applied (default: :obj:`None`).
    """

    supported_types = {ColType.NUMERICAL}

    def __init__(
        self,
        in_dim: int = 1,
        out_dim: int | None = None,
        stats_list: List[Dict[StatType, Any]] | None = None,
        post_module: torch.nn.Module | None = None,
        activation: torch.nn.Module | None = None,
    ):
        super().__init__(out_dim, stats_list, post_module)
        self.in_dim = in_dim
        self.activation = activation

    def post_init(self):
        r"""This is the actual initialization function."""
        mean = torch.tensor([stats[StatType.MEAN] for stats in self.stats_list])
        self.register_buffer("mean", mean)
        std = torch.tensor([stats[StatType.STD] for stats in self.stats_list]) + 1e-6
        self.register_buffer("std", std)
        num_cols = len(self.stats_list)
        self.weight = Parameter(torch.empty(num_cols, self.in_dim, self.out_dim))
        self.bias = Parameter(torch.empty(num_cols, self.out_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        super().reset_parameters()
        torch.nn.init.normal_(self.weight, std=0.01)
        torch.nn.init.zeros_(self.bias)

    def encode_forward(
        self,
        feat: Tensor,
    ) -> Tensor:
        if feat.ndim == 2:
            # feat: [batch_size, num_cols]
            feat = ((feat - self.mean) / self.std).unsqueeze(-1)
        elif feat.ndim == 3:
            # feat: [batch_size, num_cols, 1]
            feat = (feat - self.mean.unsqueeze(-1)) / self.std.unsqueeze(-1)
        # [batch_size, num_cols, in_dim], [num_cols, in_dim, out_dim]
        # -> [batch_size, num_cols, out_dim]
        scale = feat.size(-1)
        x_lin = torch.einsum("ijk,jkl->ijl", feat, self.weight) / scale
        # [batch_size, num_cols, out_dim] + [num_cols, out_dim]
        # -> [batch_size, num_cols, out_dim]
        x = x_lin + self.bias

        if self.activation is not None:
            x = self.activation(x)
        return x
