from functools import lru_cache

from torch import Tensor

from rllm.transforms.graph_transforms import EdgeTransform
from rllm.transforms.graph_transforms.functional import add_remaining_self_loops


class AddRemainingSelfLoops(EdgeTransform):
    r"""Adds missing self-loops to the adjacency matrix.

    .. math::
        \mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}

    Args:
        fill_value (Any): Value used for added self-loops.
            (default: :obj:`1.0`)
    """

    def __init__(self, fill_value=1.0):
        super().__init__()
        self.fill_value = fill_value

    @lru_cache()
    def forward(self, adj: Tensor) -> Tensor:
        return add_remaining_self_loops(adj, self.fill_value)
