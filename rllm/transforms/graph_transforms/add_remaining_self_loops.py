from typing import Any, Union

from rllm.transforms.utils import add_remaining_self_loops
from rllm.data.graph_data import GraphData, HeteroGraphData
from rllm.transforms.graph_transforms import BaseTransform


class AddRemainingSelfLoops(BaseTransform):
    r"""Add self-loops into the adjacency matrix.

    .. math::
        \mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}

    Args:
        fill_value (Any): values to be filled in the self-loops,
            the default values is 1.
    """
    def __init__(self, fill_value: Any = 1.):
        self.fill_value = fill_value

    def forward(self, data: Any):
        if isinstance(data, Union[GraphData, HeteroGraphData]):
            assert data.adj is not None
            data.adj = add_remaining_self_loops(data.adj)
        elif isinstance(data, HeteroGraphData):
            for store in data.edge_stores:
                if 'adj' not in store or not store.is_bipartite():
                    continue
                store.adj = add_remaining_self_loops(store.adj)
        return data
