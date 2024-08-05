from typing import Any, List, Dict, Optional

import torch
import torch.nn.functional as F

from rllm.types import ColType
from rllm.nn.models import TabTransformer
from rllm.nn.conv import GCNConv


class Bridge(torch.nn.Module):
    r"""Bridge method introduced in the
    `rLLM: Relational Table Learning with LLMs
    <https://arxiv.org/abs/2407.20157>`__ paper.
    Here is a simple example, any suitable TNN and GNN can be used.

    Args:
        table_hidden_dim (int): Hidden dimensionality for catrgorical features.
        table_output_dim (int): Output dimensionality for table encoder,
            as well as the input dimensionality of graph encoder.
        table_layers (int): Number of TNN layers.
        table_heads (int): Number of heads in the self-attention layer.
        graph_dropout (int): Dropout for graph encoder.
        graph_hidden_dim (int): Hidden dimensionality for GCN.
        graph_output_dim (int):  Output dimensionality for graph encoder.
        graph_layers (int): GCN layers.
        stats_dict (Dict[:class:`rllm.types.ColType`, List[dict[str, Any]]):
            A dictionary that maps column type into stats. The column
            with same :class:`rllm.types.ColType` will be put together."""

    def __init__(
        self,
        table_hidden_dim: int,
        table_output_dim: int,
        stats_dict: Dict[ColType, List[Dict[str, Any]]],
        graph_output_dim: int,
        graph_hidden_dim: Optional[int] = None,
        table_layers: int = 2,
        graph_layers: int = 1,
        table_heads: int = 8,
        graph_dropout: int = 0.5,
    ):
        super().__init__()
        self.dropout = graph_dropout
        self.table_encoder = TabTransformer(
            hidden_dim=table_hidden_dim,  # embedding dimension
            output_dim=table_output_dim,  # multi-class prediction
            layers=table_layers,  # depth
            heads=table_heads,  # heads
            col_stats_dict=stats_dict,
        )

        layers = []
        if graph_layers >= 2:
            layers.append(GCNConv(table_output_dim, graph_hidden_dim))
            for _ in range(graph_layers - 2):
                layers.append(GCNConv(graph_hidden_dim, graph_hidden_dim))
            layers.append(GCNConv(graph_hidden_dim, graph_output_dim))
        else:
            layers.append(GCNConv(table_output_dim, graph_output_dim))
        self.graph_encoder = torch.nn.ModuleList(layers)

    def forward(self, table, x, adj, valid, total):
        feat_dict = table.get_feat_dict()  # A dict contains feature tensor.
        x_valid = self.table_encoder(feat_dict)
        x = torch.cat([x_valid, x[valid:total, :]], dim=0)

        for layer in self.graph_encoder[:-1]:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = layer(x, adj)
            x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.graph_encoder[-1](x, adj)
        return x[:valid, :]  # Only return valid sample embedding.
