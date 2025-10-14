from functools import lru_cache

from torch import Tensor

from rllm.transforms.graph_transforms import NodeTransform
from rllm.transforms.graph_transforms.functional import svd_feature_reduction


class SVDFeatureReduction(NodeTransform):
    r"""Dimensionality reduction of node features via Singular Value
    Decomposition (SVD).

    Args:
        out_dim (int): The dimensionlity of node features after
            reduction.
    """

    def __init__(
        self,
        out_dim: int,
    ):
        self.out_dim = out_dim

    @lru_cache()
    def forward(self, x: Tensor) -> Tensor:
        return svd_feature_reduction(x, self.out_dim)
