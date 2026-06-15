from typing import TYPE_CHECKING

from rllm.utils.lazy_imports import define_lazy_imports


if TYPE_CHECKING:
    from rllm.nn.attention.global_attn import (
        GlobalAttn,
        VectorQuantizerEMA,
    )

_LAZY_MODULES = {
    "rllm.nn.attention.global_attn": (
        "GlobalAttn",
        "VectorQuantizerEMA",
    ),
}

__all__, __getattr__, __dir__ = define_lazy_imports(
    __name__,
    globals(),
    _LAZY_MODULES,
)
