from typing import TYPE_CHECKING

from rllm.utils.lazy_imports import define_lazy_imports


if TYPE_CHECKING:
    from rllm.nn.encoder.col_encoder._embedding_encoder import EmbeddingEncoder
    from rllm.nn.encoder.col_encoder._linear_encoder import LinearEncoder

_LAZY_MODULES = {
    "rllm.nn.encoder.col_encoder._embedding_encoder": (
        "EmbeddingEncoder",
    ),
    "rllm.nn.encoder.col_encoder._linear_encoder": (
        "LinearEncoder",
    ),
}

__all__, __getattr__, __dir__ = define_lazy_imports(
    __name__,
    globals(),
    _LAZY_MODULES,
)
