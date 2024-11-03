from __future__ import annotations
from typing import Any, Dict, List
import torch
from torch import Tensor
from torch.nn import (
    Embedding,
    Module,
    Parameter,
)

from rllm.types import ColType, NAMode, StatType
from rllm.transforms.table_transforms import ColTypeTransform


class EmbeddingEncoder(ColTypeTransform):
    r"""An simple embedding look-up based Transform for categorical features.
    It applies :class:`torch.nn.Embedding` for each categorical feature and
    concatenates the output embeddings.
    """

    supported_types = {ColType.CATEGORICAL}

    def __init__(
        self,
        out_dim: int | None = None,
        stats_list: List[Dict[StatType, Any]] | None = None,
        col_type: ColType | None = ColType.CATEGORICAL,
        post_module: Module | None = None,
        na_mode: NAMode | None = None,
    ) -> None:
        super().__init__(out_dim, stats_list, col_type, post_module, na_mode)

    def post_init(self):
        r"""This is the actual initialization function."""
        num_categories_list = [0]
        for stats in self.stats_list:
            num_categories = stats[StatType.COUNT]
            num_categories_list.append(num_categories)
        # Single embedding module that stores embeddings of all categories
        # across all categorical columns.
        # 0-th category is for NaN.
        self.emb = Embedding(
            sum(num_categories_list) + 1,
            self.out_dim,
            padding_idx=0,
        )
        # [num_cols, ]
        self.register_buffer(
            "offset",
            torch.cumsum(
                torch.tensor(num_categories_list[:-1], dtype=torch.long), dim=0
            ),
        )
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.emb.reset_parameters()

    def encode_forward(
        self,
        feat: Tensor,
        col_names: List[str] | None = None,
    ) -> Tensor:
        # feat: [batch_size, num_cols]
        # Get NaN mask
        na_mask = feat < 0
        # Increment the index by one not to conflict with the padding idx
        # Also add offset for each column to avoid embedding conflict
        feat = feat + self.offset + 1
        # Use 0th index for NaN
        feat[na_mask] = 0
        # [batch_size, num_cols, dim]
        return self.emb(feat)


class LinearEncoder(ColTypeTransform):
    r"""A linear function based Transform for numerical features. It applies
    linear layer :obj:`torch.nn.Linear(1, out_dim)` on each raw numerical
    feature and concatenates the output embeddings. Note that the
    implementation does this for all numerical features in a batched manner.
    """

    supported_types = {ColType.NUMERICAL}

    def __init__(
        self,
        out_dim: int | None = None,
        stats_list: List[Dict[StatType, Any]] | None = None,
        col_type: ColType | None = ColType.NUMERICAL,
        post_module: Module | None = None,
        na_mode: NAMode | None = None,
    ):
        super().__init__(out_dim, stats_list, col_type, post_module, na_mode)

    def post_init(self):
        r"""This is the actual initialization function."""
        mean = torch.tensor([stats[StatType.MEAN] for stats in self.stats_list])
        self.register_buffer("mean", mean)
        std = torch.tensor([stats[StatType.STD] for stats in self.stats_list]) + 1e-6
        self.register_buffer("std", std)
        num_cols = len(self.stats_list)
        self.weight = Parameter(torch.empty(num_cols, self.out_dim))
        self.bias = Parameter(torch.empty(num_cols, self.out_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        super().reset_parameters()
        torch.nn.init.normal_(self.weight, std=0.01)
        torch.nn.init.zeros_(self.bias)

    def encode_forward(
        self,
        feat: Tensor,
        col_names: List[str] | None = None,
    ) -> Tensor:
        # feat: [batch_size, num_cols]
        feat = (feat - self.mean) / self.std
        # [batch_size, num_cols], [dim, num_cols]
        # -> [batch_size, num_cols, dim]
        x_lin = torch.einsum("ij,jk->ijk", feat, self.weight)
        # [batch_size, num_cols, dim] + [num_cols, dim]
        # -> [batch_size, num_cols, dim]
        x = x_lin + self.bias
        return x


class StackEncoder(ColTypeTransform):
    r"""Simply stack input numerical features of shape
    :obj:`[batch_size, num_cols]` into
    :obj:`[batch_size, num_cols, out_dim]`.
    """

    supported_types = {ColType.NUMERICAL}

    def __init__(
        self,
        out_dim: int | None = None,
        stats_list: List[Dict[StatType, Any]] | None = None,
        col_type: ColType | None = ColType.NUMERICAL,
        post_module: Module | None = None,
        na_mode: NAMode | None = None,
    ) -> None:
        super().__init__(out_dim, stats_list, col_type, post_module, na_mode)

    def post_init(self) -> None:
        mean = torch.tensor([stats[StatType.MEAN] for stats in self.stats_list])
        self.register_buffer("mean", mean)
        std = torch.tensor([stats[StatType.STD] for stats in self.stats_list]) + 1e-6
        self.register_buffer("std", std)

    def reset_parameters(self) -> None:
        super().reset_parameters()

    def encode_forward(
        self,
        feat: Tensor,
        col_names: List[str] | None = None,
    ) -> Tensor:
        # feat: [batch_size, num_cols]
        feat = (feat - self.mean) / self.std
        # x: [batch_size, num_cols, out_dim]
        x = feat.unsqueeze(2).repeat(1, 1, self.out_dim)
        return x
