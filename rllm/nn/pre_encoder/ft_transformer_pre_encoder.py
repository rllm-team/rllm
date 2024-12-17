from __future__ import annotations
from typing import Any, Dict, List

from .pre_encoder import PreEncoder
from ._embedding_encoder import EmbeddingEncoder
from ._linear_encoder import LinearEncoder
from rllm.types import ColType


class FTTransformerPreEncoder(PreEncoder):
    def __init__(
        self,
        out_dim: int,
        metadata: Dict[ColType, List[Dict[str, Any]]],
        in_dim: int = 1,
    ) -> None:
        col_pre_encoder_dict = {
            ColType.CATEGORICAL: EmbeddingEncoder(),
            ColType.NUMERICAL: LinearEncoder(in_dim=in_dim),
        }
        super().__init__(out_dim, metadata, col_pre_encoder_dict)
