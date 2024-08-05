from typing import Optional, Callable
import pandas as pd
import torch
from torch import Tensor

from rllm.data import GraphData


def build_homo_graph(
    df: pd.DataFrame,
    n_src: int,
    n_tgt: int,
    x: Tensor,
    y: Optional[Tensor] = None,
    transform: Optional[Callable] = None,
):
    r"""Use the given dataframe to construct a simple undirected and
        unweighted graph with only two types of nodes and one type of edge.

    Args:
        df (pd.DataFrame): The given dataframe, where the first two columns
            represent two types of nodes that may be connected in a
            homogeneous graph (abbreviated as sorce nodes and target nodes).
            Assume that the indices of source and target nodes
            in the dataframe start from 0.
        src_nodes (int): Total amount of source nodes.
        tgt_nodes (int): Total amount of target nodes.
        x (Tensor): Features of nodes.
        y (Optional[Tensor]): Labels of (part) nodes.
        names (List[str]):
            The names of the two types of nodes in the generated graph.
        transform (Optional[Callable]):
            A function/transform that takes in a :obj:`GraphData`
            and returns a transformed version.
    """

    n_all = n_src + n_tgt
    assert n_all == x.size(0)

    # Get adj
    src_nodes, tgt_nodes = torch.from_numpy(df.iloc[:, :2].values).t()
    indices = torch.cat(
        [
            torch.stack([src_nodes, tgt_nodes], dim=0),  # src -> tgt
            torch.stack([tgt_nodes, src_nodes], dim=0),  # tgt -> src
        ],
        dim=1,
    )
    values = torch.ones((indices.shape[1],), dtype=torch.float32)
    adj = torch.sparse_coo_tensor(indices, values, (n_all, n_all))

    # Construct graph
    graph = GraphData(x=x, y=y, adj=adj)
    # Use transform
    if transform:
        graph = transform(graph)

    return graph
