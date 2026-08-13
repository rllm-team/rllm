from typing import TYPE_CHECKING

from rllm.utils.lazy_imports import define_lazy_imports


if TYPE_CHECKING:
    from rllm.dataloader.sampler.data_type import (
        NodeSamplerInput,
        HeteroSamplerOutput,
    )
    from rllm.dataloader.sampler.hetero_sampler import HeteroSampler

_LAZY_MODULES = {
    "rllm.dataloader.sampler.data_type": (
        "NodeSamplerInput",
        "HeteroSamplerOutput",
    ),
    "rllm.dataloader.sampler.hetero_sampler": (
        "HeteroSampler",
    ),
}

__all__, __getattr__, __dir__ = define_lazy_imports(
    __name__,
    globals(),
    _LAZY_MODULES,
)
