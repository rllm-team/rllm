from typing import Union
from functools import lru_cache

from torch import Tensor

from rllm.data.graph_data import GraphData, HeteroGraphData
from rllm.transforms.graph_transforms import NETransform
from rllm.transforms.graph_transforms.functional import (
    add_remaining_self_loops_,
    symmetric_norm,
)


class GCNNorm(NETransform):
    r"""Normalize the sparse adjacency matrix from the `"Semi-supervised
    Classification with GraphConvolutional
    Networks" <https://arxiv.org/abs/1609.02907>`__ .

    .. math::
        \mathbf{\hat{A}} = \mathbf{\hat{D}}^{-1/2} (\mathbf{A} + \mathbf{I})
        \mathbf{\hat{D}}^{-1/2}
    """

    def __init__(self):
        self.data = None

    @lru_cache(maxsize=128)
    def forward(self, data: Union[Tensor, GraphData, HeteroGraphData]):
        # ZK: for batchlize training, we do not use cache here.
        # if self.data is not None:
        #     return self.data

        if isinstance(data, GraphData):
            assert data.adj is not None
            data.adj = self.gcn_norm(data.adj)
        elif isinstance(data, HeteroGraphData):
            if "adj" in data:
                data.adj = self.gcn_norm(data.adj)
            for store in data.edge_stores:
                if "adj" not in store or store.is_bipartite():
                    continue
                data.adj = self.gcn_norm(data.adj)
        elif isinstance(data, Tensor):
            assert data.size(0) == data.size(1)
            data = self.gcn_norm(data)
        self.data = data
        return data

    def gcn_norm(self, adj: Tensor):
        adj = add_remaining_self_loops_(adj)
        return symmetric_norm(adj)
