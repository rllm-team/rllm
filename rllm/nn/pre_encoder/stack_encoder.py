from __future__ import annotations
from typing import Any, Dict, List

import torch
from torch import Tensor
from torch.nn import Module

from rllm.types import ColType, NAMode, StatType
from rllm.transforms.table_transforms import ColTypeTransform


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
    ) -> Tensor:
        # feat: [batch_size, num_cols]
        feat = (feat - self.mean) / self.std
        # x: [batch_size, num_cols, out_dim]
        x = feat.unsqueeze(2).repeat(1, 1, self.out_dim)
        return x
