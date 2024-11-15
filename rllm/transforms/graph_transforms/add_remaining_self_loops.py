from typing import Union

from rllm.data.graph_data import GraphData, HeteroGraphData
from rllm.transforms.graph_transforms import BaseTransform
from rllm.transforms.graph_transforms.functional import add_remaining_self_loops


class AddRemainingSelfLoops(BaseTransform):
    r"""Add self-loops into the adjacency matrix.

    .. math::
        \mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}

    Args:
        fill_value (Any): values to be filled in the self-loops,
            the default values is 1.0
    """

    def __init__(self, fill_value=1.0):
        self.fill_value = fill_value

    def forward(self, data):
        if isinstance(data, Union[GraphData, HeteroGraphData]):
            assert data.adj is not None
            data.adj = add_remaining_self_loops(data.adj)
        elif isinstance(data, HeteroGraphData):
            for store in data.edge_stores:
                if "adj" not in store or not store.is_bipartite():
                    continue
                store.adj = add_remaining_self_loops(store.adj)
        return data
