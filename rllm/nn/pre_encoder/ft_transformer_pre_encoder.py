from __future__ import annotations
from typing import Any, Dict, List

from .pre_encoder import PreEncoder
from rllm.types import ColType
from rllm.nn.pre_encoder import EmbeddingPreEncoder, LinearPreEncoder


class FTTransformerPreEncoder(PreEncoder):
    def __init__(
        self,
        out_dim: int,
        metadata: Dict[ColType, List[Dict[str, Any]]],
        in_dim: int = 1,
    ) -> None:
        col_pre_encoder_dict = {
            ColType.CATEGORICAL: EmbeddingPreEncoder(),
            ColType.NUMERICAL: LinearPreEncoder(in_dim=in_dim),
        }
        super().__init__(out_dim, metadata, col_pre_encoder_dict)
