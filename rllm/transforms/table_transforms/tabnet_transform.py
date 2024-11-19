from __future__ import annotations
from typing import Any, Dict, List

import torch

from rllm.types import ColType, NAMode
from rllm.transforms.table_transforms import ColTypeTransform, TableTypeTransform
from rllm.nn.pre_encoder import EmbeddingEncoder, StackEncoder


class TabNetTransform(TableTypeTransform):
    def __init__(
        self,
        out_dim: int,
        col_stats_dict: Dict[ColType, List[Dict[str, Any]]],
        col_types_transform_dict: Dict[ColType, ColTypeTransform] = None,
    ) -> None:
        if col_types_transform_dict is None:
            col_types_transform_dict = {
                ColType.CATEGORICAL: EmbeddingEncoder(
                    na_mode=NAMode.MOST_FREQUENT,
                ),
                ColType.NUMERICAL: StackEncoder(
                    out_dim=out_dim,
                    na_mode=NAMode.MEAN,
                ),
            }
        self.out_dim = out_dim
        self.col_stats_dict = col_stats_dict
        self.col_types_transform_dict = col_types_transform_dict

    def post_init(self) -> None:
        super().__init__(
            self.out_dim, self.col_stats_dict, self.col_types_transform_dict
        )
