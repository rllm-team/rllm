from typing import Any, Dict, List, Type

import torch
from torch.nn import Module
import torch.nn.functional as F

from rllm.types import ColType
from rllm.nn.conv.table_conv import TabTransformerConv
from rllm.nn.conv.graph_conv import GCNConv


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


class Bridge(torch.nn.Module):
    def __init__(
        self,
        table_encoder: TableEncoder,
        graph_encoder: GraphEncoder,
    ) -> None:
        super().__init__()
        self.table_encoder = table_encoder
        self.graph_encoder = graph_encoder

    def forward(self, table, non_table, adj):
        t_embedds = self.table_encoder(table)
        node_feats = torch.cat([t_embedds, non_table], dim=0)
        node_feats = self.graph_encoder(node_feats, adj)
        return node_feats[: len(table), :]
