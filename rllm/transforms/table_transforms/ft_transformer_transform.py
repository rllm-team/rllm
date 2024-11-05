from __future__ import annotations
from typing import Any, Dict, List

from rllm.types import ColType
from rllm.transforms.table_transforms import (
    ColTypeTransform,
    TableTypeTransform,
)
from rllm.nn.pre_encoder import EmbeddingEncoder, LinearEncoder


class FTTransformerTransform(TableTypeTransform):
    def __init__(
        self,
        out_dim: int,
        col_stats_dict: Dict[ColType, List[Dict[str, Any]]],
        col_types_transform_dict: Dict[ColType, ColTypeTransform] = None,
    ) -> None:
        if col_types_transform_dict is None:
            col_types_transform_dict = {
                ColType.CATEGORICAL: EmbeddingEncoder(),
                ColType.NUMERICAL: LinearEncoder(),
            }
        super().__init__(out_dim, col_stats_dict, col_types_transform_dict)
