from __future__ import annotations
from typing import Any, Dict, List

from rllm.types import ColType
from .pre_encoder import PreEncoder
from rllm.nn.pre_encoder import EmbeddingEncoder, NumericalDefaultEncoder


class TabTransformerEncoder(PreEncoder):
    def __init__(
        self,
        out_dim: int,
        metadata: Dict[ColType, List[Dict[str, Any]]] = None,
    ) -> None:
        self.col_types_transform_dict = {
            ColType.CATEGORICAL: EmbeddingEncoder(
                out_dim=out_dim,
                stats_list=(
                    metadata[ColType.CATEGORICAL] if metadata is not None else None
                ),
            ),
            ColType.NUMERICAL: NumericalDefaultEncoder(),
        }

        super().__init__(out_dim, metadata, self.col_types_transform_dict)
