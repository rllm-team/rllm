from typing import TYPE_CHECKING

from rllm.utils.lazy_imports import define_lazy_imports


if TYPE_CHECKING:
    from rllm.dataloader.neighbor_loader import NeighborLoader
    from rllm.dataloader.bridge_loader import BRIDGELoader
    from rllm.dataloader.relbench_loader import RelbenchLoader

_LAZY_MODULES = {
    "rllm.dataloader.neighbor_loader": (
        "NeighborLoader",
    ),
    "rllm.dataloader.bridge_loader": (
        "BRIDGELoader",
    ),
    "rllm.dataloader.relbench_loader": (
        "RelbenchLoader",
    ),
}

__all__, __getattr__, __dir__ = define_lazy_imports(
    __name__,
    globals(),
    _LAZY_MODULES,
)
