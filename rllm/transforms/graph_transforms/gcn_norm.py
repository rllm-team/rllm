from torch import Tensor

from rllm.transforms.graph_transforms import EdgeTransform
from rllm.transforms.graph_transforms.functional import (
    add_remaining_self_loops,
    symmetric_norm,
)


class GCNNorm(EdgeTransform):
    r"""Normalize the sparse adjacency matrix from the `"Semi-supervised
    Classification with GraphConvolutional
    Networks" <https://arxiv.org/abs/1609.02907>`__ .

    .. math::
        \mathbf{\hat{A}} = \mathbf{\hat{D}}^{-1/2} (\mathbf{A} + \mathbf{I})
        \mathbf{\hat{D}}^{-1/2}

    Args:
        None.

    Shape:
        - Input: Sparse or dense adjacency matrix of shape
          ``[num_nodes, num_nodes]``.
        - Output: Normalized adjacency matrix of shape
          ``[num_nodes, num_nodes]``.

    Examples:
        >>> transform = GCNNorm()
        >>> adj = transform(adj)
    """

    def __init__(self):
        pass

    def forward(self, adj: Tensor) -> Tensor:
        adj = add_remaining_self_loops(adj)
        return symmetric_norm(adj)
