from importlib import import_module
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from rllm.transforms.graph_transforms.node_edge_transform import (
        NodeTransform,
        EdgeTransform,
    )
    from rllm.transforms.graph_transforms.add_remaining_self_loops import AddRemainingSelfLoops
    from rllm.transforms.graph_transforms.remove_self_loops import RemoveSelfLoops
    from rllm.transforms.graph_transforms.knn_graph import KNNGraph
    from rllm.transforms.graph_transforms.gcn_norm import GCNNorm
    from rllm.transforms.graph_transforms.gdc import GDC
    from rllm.transforms.graph_transforms.graph_transform import GraphTransform
    from rllm.transforms.graph_transforms.gcn_transform import GCNTransform
    from rllm.transforms.graph_transforms.rect_transform import RECTTransform
    from rllm.transforms.graph_transforms.normalize_features import NormalizeFeatures
    from rllm.transforms.graph_transforms.svd_feature_reduction import SVDFeatureReduction

_LAZY_MODULES = {
    "rllm.transforms.graph_transforms.node_edge_transform": (
        "NodeTransform",
        "EdgeTransform",
    ),
    "rllm.transforms.graph_transforms.add_remaining_self_loops": (
        "AddRemainingSelfLoops",
    ),
    "rllm.transforms.graph_transforms.remove_self_loops": (
        "RemoveSelfLoops",
    ),
    "rllm.transforms.graph_transforms.knn_graph": (
        "KNNGraph",
    ),
    "rllm.transforms.graph_transforms.gcn_norm": (
        "GCNNorm",
    ),
    "rllm.transforms.graph_transforms.gdc": (
        "GDC",
    ),
    "rllm.transforms.graph_transforms.graph_transform": (
        "GraphTransform",
    ),
    "rllm.transforms.graph_transforms.gcn_transform": (
        "GCNTransform",
    ),
    "rllm.transforms.graph_transforms.rect_transform": (
        "RECTTransform",
    ),
    "rllm.transforms.graph_transforms.normalize_features": (
        "NormalizeFeatures",
    ),
    "rllm.transforms.graph_transforms.svd_feature_reduction": (
        "SVDFeatureReduction",
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
