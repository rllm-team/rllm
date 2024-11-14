from .normalize_features import normalize_features  # noqa
from .svd_feature_reduction import svd_feature_reduction  # noqa
from .remove_training_classes import remove_training_classes  # noqa

from .add_remaining_self_loops import add_remaining_self_loops  # noqa
from .remove_self_loops import remove_self_loops  # noqa
from .knn_graph import knn_graph  # noqa
from .symmetric_norm import symmetric_norm  # noqa

general_func = [
    "normalize_features",
    "svd_feature_reduction",
    "remove_training_classes",
]


graph_func = [
    "add_remaining_self_loops",
    "remove_self_loops",
    "knn_graph",
    "symmetric_norm",
]

__all__ = general_func + graph_func
