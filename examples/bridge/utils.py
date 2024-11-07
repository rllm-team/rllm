from typing import Any, Dict, List, Optional, Callable
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor

from rllm.data import GraphData
from rllm.nn.conv.graph_conv.gcn_conv import GCNConv
from rllm.transforms.table_transforms import FTTransformerTransform
from rllm.nn.conv.table_conv import TabTransformerConv
from rllm.types import ColType


def get_homo_data(
    relation_df: pd.DataFrame,
    src_col_name: str,
    tgt_col_name: str,
    src_emb: Tensor,
    tgt_emb: Tensor,
):
    # Making relationship
    n_src = src_emb.size(0)
    ordered_rating = relation_df.assign(
        **{
            src_col_name: relation_df[src_col_name] - 1,
            tgt_col_name: relation_df[tgt_col_name] + n_src - 1,
        }
    )

    # Making embedding
    x = torch.cat([src_emb, tgt_emb], dim=0)
    return (x, ordered_rating)


def build_homo_graph(
    relation_df: pd.DataFrame,
    x: Tensor,
    y: Optional[Tensor] = None,
    transform: Optional[Callable] = None,
    edge_per_node: Optional[int] = None,
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
    adj = torch.sparse_coo_tensor(indices, values, (x.size(0), x.size(0)))

    # Construct graph
    graph = GraphData(x=x, y=y, adj=adj)

    # Use transform
    if transform:
        graph = transform(graph)

    return graph


class GraphEncoder(torch.nn.Module):
    r"""GraphEncoder is a submodule of the BRIDGE method,
    which mainly performs multi-layer convolution of the incoming graph.

    Args:
        in_dim (int): Size of each input sample.
        hidden_dim (int): Size of each sample in hidden layer.
        out_dim (int): Size of each output sample.
        dropout (float): Dropout probability.
        graph_conv : Using the graph convolution layer.
        num_layers (int): The number of layers of the convolution.
        activate (str):Activate the function.
        **kwargs: Parameters required for different convolution layers.
    """

    def __init__(
        self,
        in_dim,
        hidden_dim,
        out_dim,
        dropout,
        graph_conv: GCNConv,
        num_layers: int = 2,
        activate: str = "relu",
        **kwargs,
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.activate = activate
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        conv_args = kwargs["conv_params"] if "conv_params" in kwargs.keys() else None
        if num_layers >= 2:
            # First layer
            self.convs.append(graph_conv(in_dim, hidden_dim, conv_args))

            # Intermediate layer
            for _ in range(num_layers - 2):
                self.convs.append(graph_conv(hidden_dim, hidden_dim, conv_args))

            # Last layer
            self.convs.append(graph_conv(hidden_dim, out_dim, conv_args))
        else:
            # Only layer
            self.convs.append(graph_conv(in_dim, out_dim, conv_args))

    def forward(self, x, adj):
        for layer in range(self.num_layers - 1):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.relu(self.convs[layer](x, adj))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj)
        return x


class TableEncoder(torch.nn.Module):
    r"""TableEncoder is a submodule of the BRIDGE method,
    which mainly performs multi-layer convolution of the incoming table.

    Args:
        hidden_dim (int): Size of each sample in hidden layer.
        stats_dict (Dict[ColType, List[Dict[str, Any]]]):
            A dictionary that maps column type into stats.
        table_transorm: The transform method of the table.
        table_conv: Using the table convolution layer.
        num_layers (int): The number of layers of the convolution.
        **kwargs: Parameters required for different convolution layers.
    """

    def __init__(
        self,
        hidden_dim,
        stats_dict: Dict[ColType, List[Dict[str, Any]]],
        table_transorm: FTTransformerTransform,
        table_conv: TabTransformerConv,
        num_layers: int = 1,
        **kwargs,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers

        self.table_transform = table_transorm(
            out_dim=hidden_dim,
            col_stats_dict=stats_dict,
        )

        conv_args = kwargs["conv_params"] if "conv_params" in kwargs.keys() else None
        self.convs = torch.nn.ModuleList()
        for _ in range(self.num_layers):
            self.convs.append(table_conv(dim=hidden_dim, **conv_args))

    def forward(self, table):
        feat_dict = table.get_feat_dict()  # A dict contains feature tensor.
        x, _ = self.table_transform(feat_dict)
        for table_conv in self.convs:
            x = table_conv(x)
        x = x.mean(dim=1)
        return x
