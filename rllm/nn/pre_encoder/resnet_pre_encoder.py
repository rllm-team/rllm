from typing import Any, Dict, List

from rllm.types import ColType
from .pre_encoder import PreEncoder
from ._embedding_encoder import EmbeddingEncoder
from ._linear_encoder import LinearEncoder
from ._timestamp_encoder import TimestampEncoder
from ._textembedding_encoder import TextEmbeddingEncoder


class ResNetPreEncoder(PreEncoder):
    r"""The pre-encoder for ResNet TNN.

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
            ColType.NUMERICAL: LinearEncoder(),
            ColType.TIMESTAMP: TimestampEncoder(),
            ColType.TEXT: TextEmbeddingEncoder(),
        }

        super().__init__(out_dim, metadata, col_pre_encoder_dict)
