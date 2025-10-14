from functools import lru_cache

from torch import Tensor

from rllm.transforms.graph_transforms import EdgeTransform
from rllm.transforms.graph_transforms.functional import add_remaining_self_loops


class AddRemainingSelfLoops(EdgeTransform):
    r"""Add self-loops into the adjacency matrix.

    .. math::
        \mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}

    Args:
        fill_value (Any): values to be filled in the self-loops,
            the default values is 1.0
    """

    def __init__(self, fill_value=1.0):
        self.fill_value = fill_value
        self.data = None

    @lru_cache()
    def forward(self, adj: Tensor) -> Tensor:
        return add_remaining_self_loops(adj, self.fill_value)
