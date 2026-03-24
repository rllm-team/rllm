from typing import Any, Dict, List

from rllm.types import ColType
from .pre_encoder import PreEncoder
from .col_encoder._embedding_encoder import EmbeddingEncoder
from .col_encoder._linear_encoder import LinearEncoder
from .col_encoder._textembedding_encoder import TextEmbeddingEncoder
from .col_encoder._timestamp_encoder import TimestampEncoder


class ResNetPreEncoder(PreEncoder):
    r"""The pre-encoder for ResNet TNN.

    This encoder builds column-type-specific pre-encoders, then delegates the
    shared table encoding pipeline to :class:`PreEncoder`.

    Args:
        out_dim (int): The output dimensionality.
        metadata (Dict[ColType, List[Dict[str, Any]]]):
            Metadata for each column type, specifying the statistics and
            properties of the columns.

    Example:
        >>> from rllm.nn.encoder import ResNetPreEncoder
        >>> from rllm.types import ColType
        >>> metadata = {
        ...     ColType.CATEGORICAL: [{"num_classes": 100}],
        ...     ColType.NUMERICAL: [{"mean": 0.0, "std": 1.0}],
        ... }
        >>> encoder = ResNetPreEncoder(out_dim=32, metadata=metadata)
    """

    def __init__(
        self,
        out_dim: int,
        metadata: Dict[ColType, List[Dict[str, Any]]],
    ) -> None:
        # Select one col-encoder per column type.
        col_encoder_dict = {
            ColType.CATEGORICAL: [EmbeddingEncoder()],
            ColType.NUMERICAL: [LinearEncoder()],
            ColType.TIMESTAMP: [TimestampEncoder()],
            ColType.TEXT: [TextEmbeddingEncoder()],
        }

        super().__init__(out_dim, metadata, col_encoder_dict)
