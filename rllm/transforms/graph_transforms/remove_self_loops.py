from functools import lru_cache

from torch import Tensor

from rllm.transforms.graph_transforms import EdgeTransform
from rllm.transforms.graph_transforms.functional import remove_self_loops


class RemoveSelfLoops(EdgeTransform):
    r"""Removes diagonal self-loop edges from an adjacency matrix."""

    def __init__(self):
        super().__init__()

    @lru_cache()
    def forward(self, adj: Tensor) -> Tensor:
        return remove_self_loops(adj)
