from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn

from typing import List, Union

from rllm.data import TableData
from rllm.nn.encoder import ColATE
from rllm.nn.conv.graph_conv import GCNConv


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


class IntraTableInteraction(nn.Module):
    r"""InRTL intra-table interaction module.

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


class InterTableInteraction(nn.Module):
    r"""InRTL inter-table interaction module.

    This module corresponds to the HGNN simplification in Section 3.3.2 of
    the InRTL paper. Its current compatibility implementation preserves the
    existing GCN message-passing path and accepts the same homogeneous sparse
    adjacency representation used by the example script.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.5,
        num_layers: int = 2,
        graph_conv: type[nn.Module] = GCNConv,
        norm: bool = False,
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.convs.append(
                graph_conv(in_dim=in_channels, out_dim=in_channels, normalize=norm)
            )
        self.convs.append(
            graph_conv(in_dim=in_channels, out_dim=out_channels, normalize=norm)
        )

    def reset_parameters(self) -> None:
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x: Tensor, adj: Union[Tensor, List[Tensor]]) -> Tensor:
        if isinstance(adj, Tensor):
            for conv in self.convs[:-1]:
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = F.relu(conv(x, adj))
            x = F.dropout(x, p=self.dropout, training=self.training)
            return self.convs[-1](x, adj)

        for i, conv in enumerate(self.convs[:-1]):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.relu(conv(x, adj[-i - 1]))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, adj[0])


class InRTL(nn.Module):
    r"""InRTL model with direct intra-table and inter-table components.

    The model owns three paper-aligned stages: ColATE from Section 3.1, the
    linear-attention intra-table encoder from
    Section 3.3.1, and the HGNN inter-table encoder from Section 3.3.2.
    Current defaults preserve the existing example's homogeneous graph path.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        table_metadata=None,
        table_num_layers: int = 1,
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
        self.table_encoder = (
            ColATE(
                channels=in_channels,
                out_channels=in_channels,
                num_layers=table_num_layers,
                metadata=table_metadata,
            )
            if table_metadata is not None
            else None
        )
        self.intra_interaction = IntraTableInteraction(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=attn_num_layers,
            num_heads=attn_num_heads,
            beta=beta,
            dropout=attn_dropout,
        )
        self.inter_interaction = InterTableInteraction(
            in_channels=in_channels,
            out_channels=hidden_channels,
            dropout=gnn_dropout,
            num_layers=gnn_num_layers,
        )
        self.alpha = alpha
        self.aggregate = aggregate
        self.use_orig_x = use_orig_x

        if aggregate == "add":
            self.output_head = nn.Linear(hidden_channels, out_channels)
        elif aggregate == "concat":
            factor = 3 if use_orig_x else 2
            self.output_head = nn.Linear(factor * hidden_channels, out_channels)
        else:
            raise ValueError(f"Unsupported aggregation mode: {aggregate}")

    @property
    def uses_table_encoder(self) -> bool:
        return self.table_encoder is not None

    def forward_features(self, x: Tensor, adj: Union[Tensor, List[Tensor]]) -> Tensor:
        """Fuse the current intra-table and inter-table representations."""
        x_orig = x if self.use_orig_x else None
        intra_out = self.intra_interaction(x)
        inter_out = self.inter_interaction(x, adj)

        if self.aggregate == "add":
            out = self.alpha * inter_out + (1 - self.alpha) * intra_out
            if x_orig is not None:
                out = out + x_orig
        else:
            pieces = [intra_out, inter_out]
            if x_orig is not None:
                pieces.append(x_orig)
            out = torch.cat(pieces, dim=1)

        return self.output_head(out)

    def forward(
        self,
        table_or_x: Union[TableData, Tensor],
        non_table: Tensor | None,
        adj: Union[Tensor, List[Tensor]] | None = None,
    ) -> Tensor:
        if not self.uses_table_encoder:
            if not isinstance(table_or_x, Tensor):
                raise TypeError("A tensor input is required without a table encoder.")
            return self.forward_features(table_or_x, non_table)

        if not isinstance(table_or_x, TableData):
            raise TypeError("TableData input is required when a table encoder is enabled.")
        table_embeddings = self.table_encoder(table_or_x)
        node_feats = (
            torch.cat([table_embeddings, non_table], dim=0)
            if non_table is not None
            else table_embeddings
        )
        return self.forward_features(node_feats, adj)[: len(table_or_x), :]
