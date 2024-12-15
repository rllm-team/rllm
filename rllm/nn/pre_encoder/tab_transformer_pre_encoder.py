from __future__ import annotations
from typing import Any, Dict, List

from .pre_encoder import PreEncoder
from rllm.types import ColType
from rllm.nn.pre_encoder import EmbeddingPreEncoder, DefaultPreEncoder


class TabTransformerEncoder(PreEncoder):
    def __init__(
        self,
        out_dim: int,
        metadata: Dict[ColType, List[Dict[str, Any]]],
    ) -> None:
        col_pre_encoder_dict = {
            ColType.CATEGORICAL: EmbeddingPreEncoder(),
            ColType.NUMERICAL: DefaultPreEncoder(),
        }
        super().__init__(out_dim, metadata, col_pre_encoder_dict)
