from typing import Union

from rllm.transforms.utils import gcn_norm
from rllm.data.graph_data import GraphData, HeteroGraphData
from rllm.transforms.base_transform import BaseTransform


class GCNNorm(BaseTransform):
    r"""Normalize the sparse adjacency matrix from the `"Semi-supervised
    Classification with GraphConvolutional
    Networks" <https://arxiv.org/abs/1609.02907>`__ .

    .. math::
        \mathbf{\hat{A}} = \mathbf{\hat{D}}^{-1/2} (\mathbf{A} + \mathbf{I})
        \mathbf{\hat{D}}^{-1/2}
    """
    def __init__(self, norm: str = 'noloop'):
        self.norm = norm

    def forward(self, data: Union[GraphData, HeteroGraphData]):
        if isinstance(data, GraphData):
            assert data.adj is not None
            data.adj = gcn_norm(data.adj, self.norm)
        elif isinstance(data, HeteroGraphData):
            if 'adj' in data:
                data.adj = gcn_norm(data.adj, self.norm)
            for store in data.edge_stores:
                if 'adj' not in store or store.is_bipartite():
                    continue
                store.adj = gcn_norm(store.adj, self.norm)
        return data
