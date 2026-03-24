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

    Shape:
        - Input: Node feature matrix ``[num_nodes, num_features]``.
        - Output: Normalized feature matrix with same shape.

    Examples:
        >>> transform = NormalizeFeatures("l2")
        >>> x = transform(x)
    """

    def __init__(self, norm: str = "l2"):
        self.norm = norm

    def forward(self, x: Tensor) -> Tensor:
        return normalize_features(x, self.norm)
