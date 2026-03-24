from torch import Tensor

from rllm.transforms.graph_transforms import NodeTransform
from rllm.transforms.graph_transforms.functional import svd_feature_reduction


class SVDFeatureReduction(NodeTransform):
    r"""Dimensionality reduction of node features via Singular Value
    Decomposition (SVD).

    Args:
        out_dim (int): The dimensionlity of node features after
            reduction.

    Shape:
        - Input: Node feature matrix ``[num_nodes, in_dim]``.
        - Output: Node feature matrix ``[num_nodes, min(in_dim, out_dim)]``.

    Examples:
        >>> transform = SVDFeatureReduction(out_dim=128)
        >>> x = transform(x)
    """

    def __init__(
        self,
        out_dim: int,
    ):
        self.out_dim = out_dim

    def forward(self, x: Tensor) -> Tensor:
        return svd_feature_reduction(x, self.out_dim)
