from typing import Union

from rllm.data.graph_data import GraphData, HeteroGraphData
from rllm.transforms.graph_transforms import BaseTransform
from rllm.transforms.graph_transforms.functional import remove_self_loops


class RemoveSelfLoops(BaseTransform):
    r"""Remove self-loops from the adjacency matrix."""

    def __init__(self):
        pass

    def forward(self, data: Union[GraphData, HeteroGraphData]):
        if isinstance(data, GraphData):
            assert data.adj is not None
            data.adj = remove_self_loops(data.adj)
        elif isinstance(data, HeteroGraphData):
            for store in data.edge_stores:
                if "adj" not in store or not store.is_bipartite():
                    continue
                store.adj = remove_self_loops(store.adj)
        return data
