from typing import TYPE_CHECKING

from rllm.utils.lazy_imports import define_lazy_imports


if TYPE_CHECKING:
    from rllm.transforms.graph_transforms.functional.add_remaining_self_loops import add_remaining_self_loops
    from rllm.transforms.graph_transforms.functional.remove_self_loops import remove_self_loops
    from rllm.transforms.graph_transforms.functional.knn_graph import knn_graph
    from rllm.transforms.graph_transforms.functional.symmetric_norm import symmetric_norm
    from rllm.transforms.graph_transforms.functional.normalize_features import normalize_features
    from rllm.transforms.graph_transforms.functional.svd_feature_reduction import svd_feature_reduction

_LAZY_MODULES = {
    "rllm.transforms.graph_transforms.functional.add_remaining_self_loops": (
        "add_remaining_self_loops",
    ),
    "rllm.transforms.graph_transforms.functional.remove_self_loops": (
        "remove_self_loops",
    ),
    "rllm.transforms.graph_transforms.functional.knn_graph": (
        "knn_graph",
    ),
    "rllm.transforms.graph_transforms.functional.symmetric_norm": (
        "symmetric_norm",
    ),
    "rllm.transforms.graph_transforms.functional.normalize_features": (
        "normalize_features",
    ),
    "rllm.transforms.graph_transforms.functional.svd_feature_reduction": (
        "svd_feature_reduction",
    ),
}

__all__, __getattr__, __dir__ = define_lazy_imports(
    __name__,
    globals(),
    _LAZY_MODULES,
)
