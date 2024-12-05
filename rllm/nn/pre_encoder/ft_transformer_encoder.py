from __future__ import annotations
from typing import Any, Dict, List

from rllm.types import ColType
from .coltype_encoder import ColTypeEncoder
from .pre_encoder import PreEncoder
from rllm.nn.pre_encoder import EmbeddingEncoder, LinearEncoder


class FTTransformerEncoder(PreEncoder):
    def __init__(
        self,
        out_dim: int = None,
        metadata: Dict[ColType, List[Dict[str, Any]]] = None,
        col_types_transform_dict: Dict[ColType, ColTypeEncoder] = None,
    ) -> None:
        if col_types_transform_dict is None:
            col_types_transform_dict = {
                ColType.CATEGORICAL: EmbeddingEncoder(),
                ColType.NUMERICAL: LinearEncoder(),
            }

            super().__init__(out_dim, metadata, col_types_transform_dict)
