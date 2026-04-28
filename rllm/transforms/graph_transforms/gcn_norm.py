from functools import lru_cache

from torch import Tensor

from rllm.transforms.graph_transforms import EdgeTransform
from rllm.transforms.graph_transforms.functional import (
    add_remaining_self_loops,
    symmetric_norm,
)


class GCNNorm(EdgeTransform):
    r"""Applies the standard GCN adjacency normalization.

    Proposed in `"Semi-supervised Classification with Graph Convolutional
    Networks" <https://arxiv.org/abs/1609.02907>`__.

    .. math::
        \mathbf{\hat{A}} = \mathbf{\hat{D}}^{-1/2} (\mathbf{A} + \mathbf{I})
        \mathbf{\hat{D}}^{-1/2}
    """

    def __init__(self):
        super().__init__()

    @lru_cache()
    def forward(self, adj: Tensor) -> Tensor:
        adj = add_remaining_self_loops(adj)
        return symmetric_norm(adj)
