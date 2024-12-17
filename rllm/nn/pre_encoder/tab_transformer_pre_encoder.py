from __future__ import annotations
from typing import Any, Dict, List

from .pre_encoder import PreEncoder
from ._embedding_encoder import EmbeddingPreEncoder
from ._reshape_encoder import ReshapeEncoder
from rllm.types import ColType


class TabTransformerPreEncoder(PreEncoder):
    def __init__(
        self,
        out_dim: int,
        metadata: Dict[ColType, List[Dict[str, Any]]],
    ) -> None:
        col_pre_encoder_dict = {
            ColType.CATEGORICAL: EmbeddingPreEncoder(),
            ColType.NUMERICAL: ReshapeEncoder(),
        }
        super().__init__(out_dim, metadata, col_pre_encoder_dict)
