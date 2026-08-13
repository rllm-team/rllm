from typing import TYPE_CHECKING

from rllm.utils.lazy_imports import define_lazy_imports


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

__all__, __getattr__, __dir__ = define_lazy_imports(
    __name__,
    globals(),
    _LAZY_MODULES,
)
