from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn

from rllm.data import TableData
from rllm.nn.models.bridge import GraphEncoder, TableEncoder


class InRTLLinearAttentionLayer(nn.Module):
    r"""Feature-only linear self-attention used by the legacy SJTU table
    experiments.

    The layer performs a dense all-to-all aggregation over node features with
    an :math:`O(ND^2)` linear-attention style approximation and averages the
    outputs across heads.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_heads: int,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.q_proj = nn.Linear(in_channels, out_channels * num_heads)
        self.k_proj = nn.Linear(in_channels, out_channels * num_heads)
        self.v_proj = nn.Linear(in_channels, out_channels * num_heads)

        self.out_channels = out_channels
        self.num_heads = num_heads
        self.eps = eps

    def reset_parameters(self) -> None:
        self.q_proj.reset_parameters()
        self.k_proj.reset_parameters()
        self.v_proj.reset_parameters()

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        num_nodes = query.size(0)
        num_heads = self.num_heads
        head_dim = self.out_channels

        q = self.q_proj(query).reshape(-1, num_heads, head_dim)
        k = self.k_proj(key).reshape(-1, num_heads, head_dim)
        v = self.v_proj(value).reshape(-1, num_heads, head_dim)

        q = q / q.norm(p=2, dim=-1, keepdim=True).clamp_min(self.eps)
        k = k / k.norm(p=2, dim=-1, keepdim=True).clamp_min(self.eps)

        sum_k = k.sum(dim=0)
        denom = torch.einsum("nhd,hd->nh", q, sum_k).unsqueeze(-1)
        denom = denom + num_nodes

        ctx = torch.einsum("nhd,nhe->hde", k, v)
        agg = torch.einsum("nhd,hde->nhe", q, ctx)
        agg = agg + v.sum(dim=0, keepdim=True)

        return (agg / denom).mean(dim=1)


class InRTLIntraTableEncoder(nn.Module):
    r"""InRTL intra-table encoder.

    This stack applies linear attention over table-side representations and
    serves as the intra-table modeling branch.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int = 1,
        num_heads: int = 4,
        beta: float = 0.5,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.fc = nn.Linear(in_channels, hidden_channels)
        self.bns = nn.ModuleList([nn.LayerNorm(hidden_channels)])
        self.attns = nn.ModuleList()

        for _ in range(num_layers):
            self.attns.append(
                InRTLLinearAttentionLayer(
                    hidden_channels,
                    hidden_channels,
                    num_heads=num_heads,
                )
            )
            self.bns.append(nn.LayerNorm(hidden_channels))

        self.dropout = dropout
        self.activation = nn.LeakyReLU(negative_slope=0.01)
        self.beta = beta

    def reset_parameters(self) -> None:
        self.fc.reset_parameters()
        for attn in self.attns:
            attn.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc(x)
        x = self.bns[0](x)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        prev_x = x
        for i, attn in enumerate(self.attns):
            x = attn(x, x, x)
            x = self.beta * x + (1 - self.beta) * prev_x
            x = self.bns[i + 1](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            prev_x = x

        return x


class InRTLBackbone(nn.Module):
    r"""Core InRTL backbone.

    It combines the intra-table linear-attention branch with the inter-table
    graph neural network branch, then fuses their outputs by addition or
    concatenation.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        attn_num_layers: int = 1,
        attn_num_heads: int = 4,
        attn_dropout: float = 0.5,
        gnn_num_layers: int = 2,
        gnn_dropout: float = 0.5,
        alpha: float = 0.8,
        beta: float = 0.5,
        aggregate: str = "add",
        use_orig_x: bool = False,
    ) -> None:
        super().__init__()
        self.attn = InRTLIntraTableEncoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=attn_num_layers,
            num_heads=attn_num_heads,
            beta=beta,
            dropout=attn_dropout,
        )
        self.gnn = GraphEncoder(
            in_dim=in_channels,
            out_dim=hidden_channels,
            dropout=gnn_dropout,
            num_layers=gnn_num_layers,
        )
        self.alpha = alpha
        self.aggregate = aggregate
        self.use_orig_x = use_orig_x

        if aggregate == "add":
            self.fc = nn.Linear(hidden_channels, out_channels)
        elif aggregate == "concat":
            factor = 3 if use_orig_x else 2
            self.fc = nn.Linear(factor * hidden_channels, out_channels)
        else:
            raise ValueError(f"Unsupported aggregation mode: {aggregate}")

    def reset_parameters(self) -> None:
        self.attn.reset_parameters()
        for conv in self.gnn.convs:
            conv.reset_parameters()
        self.fc.reset_parameters()

    def forward(self, x: Tensor, adj: Tensor | list[Tensor] | None = None) -> Tensor:
        x_orig = x if self.use_orig_x else None
        attn_out = self.attn(x)
        gnn_out = self.gnn(x, adj)

        if self.aggregate == "add":
            out = self.alpha * gnn_out + (1 - self.alpha) * attn_out
            if x_orig is not None:
                out = out + x_orig
        else:
            pieces = [attn_out, gnn_out]
            if x_orig is not None:
                pieces.append(x_orig)
            out = torch.cat(pieces, dim=1)

        return self.fc(out)


class InRTL(nn.Module):
    r"""Full InRTL model with a table encoder and InRTL backbone."""

    def __init__(
        self,
        table_encoder: TableEncoder,
        graph_encoder: InRTLBackbone,
    ) -> None:
        super().__init__()
        self.table_encoder = table_encoder
        self.graph_encoder = graph_encoder

    def forward(
        self,
        table: TableData,
        non_table: Tensor | None,
        adj: Tensor | list[Tensor],
    ) -> Tensor:
        table_embeddings = self.table_encoder(table)
        if non_table is not None:
            node_feats = torch.cat([table_embeddings, non_table], dim=0)
        else:
            node_feats = table_embeddings
        node_feats = self.graph_encoder(node_feats, adj)
        return node_feats[: len(table), :]
