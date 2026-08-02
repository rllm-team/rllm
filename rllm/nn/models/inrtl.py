from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn

from typing import List, Union

from rllm.data import TableData
from rllm.nn.encoder import ColATE
from rllm.nn.conv.graph_conv import GCNConv


class LinearAttentionLayer(nn.Module):
    r"""Apply multi-head linear attention to node representations.

    Args:
        in_channels (int): Input feature dimension.
        out_channels (int): Output feature dimension per attention head.
        num_heads (int): Number of attention heads.
        eps (float): Minimum value used when normalizing query and key vectors.
            (default: ``1e-12``)
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
        r"""Compute attention output.

        Args:
            query (Tensor): Query features of shape
                :obj:`[num_nodes, in_channels]`.
            key (Tensor): Key features of shape
                :obj:`[num_nodes, in_channels]`.
            value (Tensor): Value features of shape
                :obj:`[num_nodes, in_channels]`.

        Returns:
            Tensor: Output features of shape :obj:`[num_nodes, out_channels]`.
        """
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
    r"""Apply stacked linear attention to node representations.

    Args:
        in_channels (int): Input feature dimension.
        hidden_channels (int): Hidden feature dimension.
        num_layers (int): Number of attention layers. (default: ``1``)
        num_heads (int): Number of attention heads per layer. (default: ``4``)
        beta (float): Residual interpolation weight. (default: ``0.5``)
        dropout (float): Dropout probability. (default: ``0.5``)
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
                LinearAttentionLayer(
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
        r"""Apply intra-table interaction.

        Args:
            x (Tensor): Input features of shape
                :obj:`[num_nodes, in_channels]`.

        Returns:
            Tensor: Output features of shape
            :obj:`[num_nodes, hidden_channels]`.
        """
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
    r"""Apply graph convolutions to node representations.

    The module accepts either one sparse adjacency shared by all layers or a
    list of sparse adjacencies, one for each layer.

    Args:
        in_channels (int): Input feature dimension.
        out_channels (int): Output feature dimension.
        dropout (float): Dropout probability. (default: ``0.5``)
        num_layers (int): Number of graph-convolution layers. (default: ``2``)
        graph_conv (type[nn.Module]): Graph-convolution module to instantiate.
            (default: :class:`~rllm.nn.conv.graph_conv.GCNConv`)
        norm (bool): Whether to enable graph-convolution normalization.
            (default: ``False``)
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
        r"""Apply graph convolution.

        Args:
            x (Tensor): Input features of shape
                :obj:`[num_nodes, in_channels]`.
            adj (Tensor or List[Tensor]): Sparse adjacency shared by all layers,
                or one sparse adjacency for each layer.

        Returns:
            Tensor: Output features of shape
            :obj:`[num_nodes, out_channels]`.
        """
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
    r"""Combine table encoding, linear attention, and graph convolution.

    When ``table_metadata`` is provided, the model encodes ``TableData`` and
    concatenates the resulting row embeddings with ``non_table`` features.
    Without table metadata, it operates directly on node features.

    Args:
        in_channels (int): Input node feature dimension.
        hidden_channels (int): Hidden feature dimension.
        out_channels (int): Number of output channels.
        table_metadata (dict, optional): Metadata used to construct ColATE.
            When :obj:`None`, table encoding is disabled. (default: :obj:`None`)
        table_num_layers (int): Number of residual layers in ColATE.
            (default: ``1``)
        attn_num_layers (int): Number of linear-attention layers.
            (default: ``1``)
        attn_num_heads (int): Number of attention heads. (default: ``4``)
        attn_dropout (float): Dropout probability in linear attention.
            (default: ``0.5``)
        gnn_num_layers (int): Number of graph-convolution layers.
            (default: ``2``)
        gnn_dropout (float): Dropout probability in graph convolution.
            (default: ``0.5``)
        alpha (float): Inter-table feature weight for additive aggregation.
            (default: ``0.8``)
        beta (float): Residual interpolation weight in linear attention.
            (default: ``0.5``)
        aggregate (str): Feature aggregation mode, either :obj:`"add"` or
            :obj:`"concat"`. (default: :obj:`"add"`)
        use_orig_x (bool): Whether to include input features in aggregation.
            (default: ``False``)
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
        r"""Fuse intra-table and inter-table representations.

        Args:
            x (Tensor): Node features of shape :obj:`[num_nodes, in_channels]`.
            adj (Tensor or List[Tensor]): Sparse adjacency input for graph
                convolution.

        Returns:
            Tensor: Predictions of shape :obj:`[num_nodes, out_channels]`.
        """
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
        r"""Compute node predictions.

        Args:
            table_or_x (TableData or Tensor): Input table when table encoding
                is enabled, otherwise node features of shape
                :obj:`[num_nodes, in_channels]`.
            non_table (Tensor, optional): Features of non-table nodes when
                table encoding is enabled. When table encoding is disabled,
                this argument is interpreted as the adjacency input.
            adj (Tensor or List[Tensor], optional): Sparse adjacency input when
                table encoding is enabled.

        Returns:
            Tensor: Predictions for table rows when table encoding is enabled;
            otherwise predictions for all input nodes.
        """
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
