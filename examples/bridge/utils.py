from typing import Any, Dict, List, Optional, Type

import pandas as pd
import torch
from torch import Tensor
from torch.nn import Module
import torch.nn.functional as F

from rllm.types import ColType
from rllm.data import GraphData
from rllm.nn.conv.table_conv import TabTransformerConv
from rllm.nn.conv.graph_conv import GCNConv


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


class TableEncoder(Module):
    r"""TableEncoder is a submodule of the BRIDGE method,
    which mainly performs multi-layer convolution of the incoming table.

    Args:
        in_dim (int): Input dimensionality of the table data.
        out_dim (int): Output dimensionality for the encoded table data.
        num_layers (int, optional): Number of convolution layers. Defaults to 1.
        table_transorm (Module): The transformation module to be applied to the table data.
        table_conv (Type[Module], optional): The convolution module to be used for
            encoding the table data. Defaults to TabTransformerConv.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_layers: int = 1,
        metadata: Dict[ColType, List[Dict[str, Any]]] = None,
        table_conv: Type[Module] = TabTransformerConv,
    ) -> None:

        super().__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(table_conv(dim=out_dim, metadata=metadata))
        for _ in range(num_layers - 1):
            self.convs.append(table_conv(dim=out_dim))

    def forward(self, table):
        x = table.feat_dict
        for conv in self.convs:
            x = conv(x)
        x = torch.cat(list(x.values()), dim=1)
        x = x.mean(dim=1)
        return x


class GraphEncoder(Module):
    r"""GraphEncoder is a submodule of the BRIDGE method,
    which mainly performs multi-layer convolution of the incoming graph.

    Args:
        hidden_dim (int): Size of each sample in hidden layer.
        out_dim (int): Size of each output sample.
        dropout (float): Dropout probability.
        num_layers (int): The number of layers of the convolution.
        graph_conv : Using the graph convolution layer.
    """

    def __init__(
        self,
        in_dim,
        out_dim,
        dropout: float = 0.5,
        num_layers: int = 2,
        graph_conv: Type[Module] = GCNConv,
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.convs = torch.nn.ModuleList()

        for _ in range(num_layers - 1):
            self.convs.append(graph_conv(in_dim=in_dim, out_dim=in_dim))
        self.convs.append(graph_conv(in_dim=in_dim, out_dim=out_dim))

    def forward(self, x, adj):
        for conv in self.convs[:-1]:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.relu(conv(x, adj))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj)
        return x
