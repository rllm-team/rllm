from __future__ import annotations
from typing import Any, Dict, List

import torch
from torch import Tensor
from torch.nn import Embedding, Module

from rllm.types import ColType, StatType
from .coltype_encoder import ColTypeEncoder


class EmbeddingEncoder(ColTypeEncoder):
    r"""An simple embedding look-up based Transform for categorical features.
    It applies :class:`torch.nn.Embedding` for each categorical feature and
    concatenates the output embeddings.
    """

    supported_types = {ColType.CATEGORICAL}

    def __init__(
        self,
        out_dim: int | None = None,
        stats_list: List[Dict[StatType, Any]] | None = None,
        post_module: Module | None = None,
    ) -> None:
        super().__init__(out_dim, stats_list, post_module)

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
