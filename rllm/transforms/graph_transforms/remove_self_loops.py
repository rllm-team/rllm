from typing import Union

from torch import Tensor

from rllm.data.graph_data import GraphData, HeteroGraphData
from rllm.transforms.graph_transforms import EdgeTransform
from rllm.transforms.graph_transforms.functional import remove_self_loops


class RemoveSelfLoops(EdgeTransform):
    r"""Remove self-loops from the adjacency matrix."""

    def __init__(self):
        self.data = None

    def forward(self, data: Union[GraphData, HeteroGraphData]):
        if self.data is not None:
            return self.data

        if isinstance(data, GraphData):
            assert data.adj is not None
            data.adj = remove_self_loops(data.adj)
        elif isinstance(data, HeteroGraphData):
            for store in data.edge_stores:
                if "adj" not in store or not store.is_bipartite():
                    continue
                store.adj = remove_self_loops(store.adj)
        elif isinstance(data, Tensor):
            assert data.size(0) == data.size(1)
            data = remove_self_loops(data)
        self.data = data
        return data
