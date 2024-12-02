from typing import Union

from torch import Tensor

from rllm.data.graph_data import GraphData, HeteroGraphData
from rllm.transforms.graph_transforms import NETransform
from rllm.transforms.graph_transforms.functional import add_remaining_self_loops


class AddRemainingSelfLoops(NETransform):
    r"""Add self-loops into the adjacency matrix.

    .. math::
        \mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}

    Args:
        fill_value (Any): values to be filled in the self-loops,
            the default values is 1.0
    """

    def __init__(self, fill_value=1.0, use_cache=False):
        self.fill_value = fill_value
        self.data = None

    def forward(self, data):
        if self.data is not None:
            return self.data

        if isinstance(data, Union[GraphData, HeteroGraphData]):
            assert data.adj is not None
            data.adj = add_remaining_self_loops(data.adj, self.fill_value)
        elif isinstance(data, HeteroGraphData):
            for store in data.edge_stores:
                if "adj" not in store or not store.is_bipartite():
                    continue
                store.adj = add_remaining_self_loops(store.adj, self.fill_value)
        elif isinstance(data, Tensor):
            assert data.size(0) == data.size(1)
            data = add_remaining_self_loops(data, self.fill_value)
        self.data = data
        return data
