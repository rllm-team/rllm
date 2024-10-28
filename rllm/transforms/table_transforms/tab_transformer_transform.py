from __future__ import annotations

from rllm.types import ColType
from rllm.transforms.table_transforms import ColTypeTransform, TableTypeTransform
from rllm.nn.pre_encoder import EmbeddingEncoder, StackEncoder

class TabTransformerTransform(TableTypeTransform):
    def __init__(
        self,
        out_dim: int,
        col_stats_dict: dict[ColType, list[dict[str,]]],
        col_types_transform_dict: dict[ColType, ColTypeTransform] = None,
    ) -> None:
        if col_types_transform_dict is None:
            col_types_transform_dict={
                ColType.CATEGORICAL: EmbeddingEncoder(),
                ColType.NUMERICAL: StackEncoder(),
            }
        super().__init__(out_dim, col_stats_dict, col_types_transform_dict)


