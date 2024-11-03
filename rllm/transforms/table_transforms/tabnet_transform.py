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
        super().__init__(out_dim, col_stats_dict, col_types_transform_dict)

    def forward(self, feat_dict):
        xs = []
        all_col_names = []
        if ColType.CATEGORICAL in self.col_stats_dict.keys():
            x_category = feat_dict[ColType.CATEGORICAL]
            category_embedding = self.transform_dict[ColType.CATEGORICAL.value](
                x_category
            )
            xs.append(category_embedding)
            col_names = self.col_names_dict[ColType.CATEGORICAL]
            all_col_names.extend(col_names)

        if ColType.NUMERICAL in self.col_stats_dict.keys():
            x_numeric = feat_dict[ColType.NUMERICAL]
            numerical_embedding = self.transform_dict[ColType.NUMERICAL.value](
                x_numeric
            )
            xs.append(numerical_embedding)
            col_names = self.col_names_dict[ColType.NUMERICAL]
            all_col_names.extend(col_names)
        x = torch.cat(xs, dim=1)
        return x, all_col_names
