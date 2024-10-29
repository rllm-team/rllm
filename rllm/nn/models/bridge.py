from typing import Any, List, Dict, Optional

import torch
import torch.nn.functional as F

from rllm.types import ColType
from rllm.transforms.table_transforms import FTTransformerTransform
from rllm.nn.conv.table_conv import TabTransformerConv
from rllm.nn.conv.graph_conv import GCNConv


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
        stats_dict: Dict[ColType, List[Dict[str, Any]]],
        graph_output_dim: int,
        graph_hidden_dim: Optional[int] = None,
        table_layers: int = 2,
        graph_layers: int = 1,
        graph_dropout: int = 0.5,
    ):
        if graph_hidden_dim is None:
            self.graph_hidden_dim = table_hidden_dim
        else:
            self.graph_hidden_dim = graph_hidden_dim
        super().__init__()
        self.dropout = graph_dropout
        self.table_transform = FTTransformerTransform(
            out_dim=table_hidden_dim,
            col_stats_dict=stats_dict,
        )
        self.table_encoder = torch.nn.ModuleList([
            TabTransformerConv(
                dim=table_hidden_dim,
            ) for _ in range(table_layers)
        ])
        # self.table_encoder = TabTransformer(
        #     hidden_dim=table_hidden_dim,  # embedding dimension
        #     output_dim=table_output_dim,  # multi-class prediction
        #     layers=table_layers,  # depth
        #     heads=table_heads,  # heads
        #     col_stats_dict=stats_dict,
        # )

        layers = []
        if graph_layers >= 2:
            layers.append(GCNConv(table_hidden_dim, self.graph_hidden_dim))
            for _ in range(graph_layers - 2):
                layers.append(GCNConv(self.graph_hidden_dim, self.graph_hidden_dim))
            layers.append(GCNConv(self.graph_hidden_dim, graph_output_dim))
        else:
            layers.append(GCNConv(table_hidden_dim, graph_output_dim))
        self.graph_encoder = torch.nn.ModuleList(layers)

    def forward(self, table, x, adj, valid, total):
        feat_dict = table.get_feat_dict()  # A dict contains feature tensor.
        x_valid, _ = self.table_transform(feat_dict)
        for layer in self.table_encoder:
            x_valid = layer(x_valid)

        x_valid = x_valid.mean(dim=1)
        x = torch.cat([x_valid, x[valid:total, :]], dim=0)

        for layer in self.graph_encoder[:-1]:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = layer(x, adj)
            x = F.relu(x)
        # Last layer without relu
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.graph_encoder[-1](x, adj)
        return x[:valid, :]  # Only return valid sample embedding.
