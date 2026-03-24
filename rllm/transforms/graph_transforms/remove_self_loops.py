from torch import Tensor

from rllm.transforms.graph_transforms import EdgeTransform
from rllm.transforms.graph_transforms.functional import remove_self_loops


class RemoveSelfLoops(EdgeTransform):
    r"""Remove self-loops from the adjacency matrix.

    Args:
        None.

    Shape:
        - Input: Sparse or dense adjacency matrix ``[num_nodes, num_nodes]``.
        - Output: Adjacency matrix of the same shape without diagonal edges.

    Examples:
        >>> transform = RemoveSelfLoops()
        >>> adj = transform(adj)
    """

    def __init__(self):
        self.data = None

    def forward(self, adj: Tensor) -> Tensor:
        return remove_self_loops(adj)
