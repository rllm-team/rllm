from typing import Optional

import pandas as pd
import torch
from torch import Tensor

from rllm.data import GraphData


def reorder_ids(
    relation_df: pd.DataFrame,
    src_col_name: str,
    tgt_col_name: str,
    n_src: int,
):
    # Making relationship
    ordered_rating = relation_df.assign(
        **{
            src_col_name: relation_df[src_col_name] - 1,
            tgt_col_name: relation_df[tgt_col_name] + n_src - 1,
        }
    )

    return ordered_rating


def build_homo_graph(
    relation_df: pd.DataFrame,
    n_all: int,
    x: Optional[Tensor] = None,
    y: Optional[Tensor] = None,
    edge_per_node: Optional[int] = None,
):
    r"""Use the given dataframe to construct a simple undirected and
        unweighted graph with only two types of nodes and one type of edge.

    Args:
        relation_df (pd.DataFrame): The given dataframe, where the first two
            columns represent two types of nodes that may be connected in a
            homogeneous graph (abbreviated as sorce nodes and target nodes).
            Assume that the indices of source and target nodes
            in the dataframe start from 0.
        n_all (int): Total amount of nodes.
        x (Tensor): Features of nodes.
        y (Optional[Tensor]): Labels of (part) nodes.
        edge_per_node (Optional[int]):
            specifying the maximum number of edges to keep for each node.
    """

    # Get adj
    src_nodes, tgt_nodes = torch.from_numpy(relation_df.iloc[:, :2].values).t()
    indices = torch.cat(
        [
            torch.stack([src_nodes, tgt_nodes], dim=0),  # src -> tgt
            torch.stack([tgt_nodes, src_nodes], dim=0),  # tgt -> src
        ],
        dim=1,
    )

    if edge_per_node is not None:
        unique_nodes = torch.unique(indices)
        mask = torch.zeros(indices.shape[1], dtype=torch.bool)

        for node in unique_nodes:
            # Find neighbors
            node_mask = (indices[0] == node) | (indices[1] == node)
            node_edges = torch.nonzero(node_mask).squeeze()

            # Randomly select `edge_per_node` edges
            if node_edges.numel() > edge_per_node:
                selected_edges = node_edges[
                    torch.randperm(node_edges.numel())[:edge_per_node]
                ]
                mask[selected_edges] = True
            else:
                mask[node_edges] = True

        indices = indices[:, mask]

    values = torch.ones((indices.shape[1],), dtype=torch.float32)
    adj = torch.sparse_coo_tensor(indices, values, (n_all, n_all))

    # Construct graph
    graph = GraphData(x=x, y=y, adj=adj)

    return graph


def accuracy_score(preds, truth):
    return (preds == truth).sum(dim=0) / len(truth)
