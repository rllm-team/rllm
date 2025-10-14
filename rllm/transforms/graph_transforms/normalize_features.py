from functools import lru_cache
from torch import Tensor

from rllm.transforms.graph_transforms import NodeTransform
from rllm.transforms.graph_transforms.functional import normalize_features


class NormalizeFeatures(NodeTransform):
    r"""Row-normalizes the node features.

    .. math::

        \vec{x} = \frac{\vec{x}}{||\vec{x}||_p}

    Args:
        norm (str): The norm to use to normalize each non zero sample,
            *e.g.*, `l1`, `l2`. (default: `l2`)
    """

    def __init__(self, norm: str = "l2"):
        self.norm = norm

    @lru_cache()
    def forward(self, x: Tensor) -> Tensor:
        return normalize_features(x, self.norm)
