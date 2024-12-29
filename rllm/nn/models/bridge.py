from typing import Any, Dict, List, Type

import torch
from torch import Tensor
import torch.nn.functional as F

from rllm.types import ColType
from rllm.nn.conv.table_conv import TabTransformerConv
from rllm.nn.conv.graph_conv import GCNConv


class TableEncoder(torch.nn.Module):
    r"""TableEncoder is a submodule of the BRIDGE method,
    which mainly performs multi-layer convolution of the incoming table.

    Args:
        in_dim (int): Input dimensionality of the table data.
        out_dim (int): Output dimensionality for the encoded table data.
        num_layers (int, optional):
            Number of convolution layers (default: :obj:`1`).
        metadata (Dict[ColType, List[Dict[str, Any]]], optional):
            Metadata for each column type, specifying the statistics and
            properties of the columns. (default: :obj:`None`).
        table_conv (Type[torch.nn.Module], optional):
            The convolution module to be used for encoding the table data
            (default: :obj:`rllm.nn.conv.table_conv.TabTransformerConv`).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_layers: int = 1,
        metadata: Dict[ColType, List[Dict[str, Any]]] = None,
        table_conv: Type[torch.nn.Module] = TabTransformerConv,
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


class GraphEncoder(torch.nn.Module):
    r"""GraphEncoder is a submodule of the BRIDGE method,
    which mainly performs multi-layer convolution of the incoming graph.

    Args:
        in_dim (int): Input dimensionality of the data.
        out_dim (int): Output dimensionality for the encoded data.
        dropout (float): Dropout probability.
        num_layers (int): The number of layers of the convolution.
        graph_conv (Type[torch.nn.Module], optional):
            The convolution module to be used for encoding the graph data
            (default: :obj:`rllm.nn.conv.graph_conv.GCNConv`).
    """

    def __init__(
        self,
        in_dim,
        out_dim,
        dropout: float = 0.5,
        num_layers: int = 2,
        graph_conv: Type[torch.nn.Module] = GCNConv,
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


class Bridge(torch.nn.Module):
    r"""The Bridge model introduced in the `"rLLM: Relational Table Learning
    with LLMs" <https://arxiv.org/abs/2407.20157>`__ paper.
    Bridge is a simple RTL method based on rLLM framework, which
    combines table neural networks (TNNs) and graph neural networks (GNNs) to
    deal with multi-table data and their interrelationships, and uses "foreign
    keys" to build relationships and analyze them to improve the performance of
    multi-table joint learning tasks.

    Args:
        table_encoder (TableEncoder): Encoder for tabular data.
        graph_encoder (GraphEncoder): Encoder for graph data.
    """

    def __init__(
        self,
        table_encoder: TableEncoder,
        graph_encoder: GraphEncoder,
    ) -> None:
        super().__init__()
        self.table_encoder = table_encoder
        self.graph_encoder = graph_encoder

    def forward(
        self,
        table: Tensor,
        non_table: Tensor,
        adj: Tensor,
    ) -> Tensor:
        """
        First, the Table Neural Network (TNN) learns the tabular data.
        Second, the learned representations are concatenated with the non-tabular data.
        Third, the Graph Neural Network (GNN) processes the combined data.
        along with the adjacency matrix to learn the overall representation.

        Args:
            table (Tensor): Input tabular data.
            non_table (Tensor): Input non-tabular data.
            adj (Tensor): Adjacency matrix.

        Returns:
            Tensor: Output node features.
        """
        t_embedds = self.table_encoder(table)
        node_feats = torch.cat([t_embedds, non_table], dim=0)
        node_feats = self.graph_encoder(node_feats, adj)
        return node_feats[: len(table), :]
