from __future__ import annotations
from typing import Any, Dict, List

from .pre_encoder import PreEncoder
from ._embedding_encoder import EmbeddingEncoder
from ._reshape_encoder import ReshapeEncoder
from rllm.types import ColType


class TabTransformerPreEncoder(PreEncoder):
    r"""The TabTransformerPreEncoder class is a specialized pre-encoder for the
    TabTransformer model. It initializes a column-specific encoder dict for
    categorical and numerical features based on the provided metadata.
    Specifically, it uses `EmbeddingEncoder` for categorical features and
    `ReshapeEncoder` for numerical features.

    Args:
        out_dim (int): The output dimensionality.
        metadata (Dict[ColType, List[Dict[str, Any]]]):
            Metadata for each column type, specifying the statistics and
            properties of the columns.
    """

    def __init__(
        self,
        out_dim: int,
        metadata: Dict[ColType, List[Dict[str, Any]]],
    ) -> None:
        col_pre_encoder_dict = {
            ColType.CATEGORICAL: EmbeddingEncoder(),
            ColType.NUMERICAL: ReshapeEncoder(),
        }
        super().__init__(out_dim, metadata, col_pre_encoder_dict)
