from .base_transform import BaseTransform  # noqa
from .compose import Compose  # noqa
from .normalize_features import NormalizeFeatures  # noqa
from .svd_feature_reduction import SVDFeatureReduction  # noqa
from .remove_training_classes import RemoveTrainingClasses  # noqa

from .add_remaining_self_loops import AddRemainingSelfLoops  # noqa
from .remove_self_loops import RemoveSelfLoops  # noqa
from .knn_graph import KNNGraph  # noqa
from .gcn_norm import GCNNorm  # noqa
from .build_homo_graph import build_homo_graph  # noqa

general_transforms = [
    'BaseTransform',
    'Compose',
    'NormalizeFeatures',
    'SVDFeatureReduction',
    'RemoveTrainingClasses',
]

graph_transforms = [
    'AddRemainingSelfLoops',
    'RemoveSelfLoops',
    'KNNGraph',
    'GCNNorm',
]

graph_builders = [
    'build_homo_graph',
]


__all__ = general_transforms + graph_transforms + graph_builders
