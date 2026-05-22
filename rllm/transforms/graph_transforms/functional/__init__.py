from importlib import import_module
from typing import TYPE_CHECKING


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

__all__ = [
    name
    for names in _LAZY_MODULES.values()
    for name in names
]

_LAZY_ATTRS = {
    name: module_name
    for module_name, names in _LAZY_MODULES.items()
    for name in names
}


def __getattr__(name):
    if name in _LAZY_ATTRS:
        module = import_module(_LAZY_ATTRS[name])
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
