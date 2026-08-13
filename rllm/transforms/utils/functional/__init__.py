from typing import TYPE_CHECKING

from rllm.utils.lazy_imports import define_lazy_imports


if TYPE_CHECKING:
    from rllm.transforms.utils.functional.remove_training_classes import remove_training_classes
    from rllm.transforms.utils.functional.metapath_prop import (
        row_norm_edge_weight,
        weighted_mean_aggregate,
        metapath_propagate,
    )

_LAZY_MODULES = {
    "rllm.transforms.utils.functional.remove_training_classes": (
        "remove_training_classes",
    ),
    "rllm.transforms.utils.functional.metapath_prop": (
        "row_norm_edge_weight",
        "weighted_mean_aggregate",
        "metapath_propagate",
    ),
}

__all__, __getattr__, __dir__ = define_lazy_imports(
    __name__,
    globals(),
    _LAZY_MODULES,
)
