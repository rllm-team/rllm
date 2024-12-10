from __future__ import annotations
from typing import Any, Dict, List

from .pre_encoder import PreEncoder
from rllm.types import ColType
from rllm.nn.pre_encoder import EmbeddingEncoder, LinearEncoder


class FTTransformerEncoder(PreEncoder):
    def __init__(
        self,
        out_dim: int,
        metadata: Dict[ColType, List[Dict[str, Any]]],
        in_dim: int = 1,
    ) -> None:
        col_encoder_dict = {
            ColType.CATEGORICAL: EmbeddingEncoder(),
            ColType.NUMERICAL: LinearEncoder(in_dim=in_dim),
        }
        super().__init__(out_dim, metadata, col_encoder_dict)
