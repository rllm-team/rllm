from .base_transform import BaseTransform
from .compose import Compose  # noqa
from .normalize_features import NormalizeFeatures  # noqa
from .svd_feature_reduction import SVDFeatureReduction  # noqa
from .remove_training_classes import RemoveTrainingClasses  # noqa
from .add_remaining_self_loops import AddRemainingSelfLoops  # noqa
from .remove_self_loops import RemoveSelfLoops  # noqa
from .knn_graph import KNNGraph  # noqa
from .gcn_norm import GCNNorm  # noqa
from .gdc import GDC


__all__ = [
    # general transforms
    "BaseTransform",
    "Compose",
    "NormalizeFeatures",
    "SVDFeatureReduction",
    "RemoveTrainingClasses",
    # graph transforms
    "AddRemainingSelfLoops",
    "RemoveSelfLoops",
    "KNNGraph",
    "GCNNorm",
    "GDC",
]
