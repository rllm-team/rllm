from typing import Union, Tuple, Optional

from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from rllm.nn.conv.graph_conv import MessagePassing
from rllm.utils import set_values


class GATConv(MessagePassing):
    dim = 0
    head_dim = 1

    def __init__(
        self,
        in_dim: Union[int, Tuple[int, int]],
        out_dim: int,
        num_heads: int = 8,
        concat: bool = False,
        negative_slope: float = 0.2,
        dropout: float = 0.6,
        edge_dim: Optional[int] = None,
        bias: bool = True,
        residual: bool = False,
        **kwargs
    ):
        super().__init__(aggr='add', **kwargs)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.edge_dim = edge_dim
        self.residual = residual

        # linear transformation
        self.add_module('lin_src', None)
        self.add_module('lin_dst', None)
        self.add_module('lin', None)
        if isinstance(in_dim, int):
            self.lin = nn.Linear(in_dim, out_dim * num_heads, bias=False)
        else:
            self.lin_src = nn.Linear(in_dim[0], out_dim * num_heads, bias=False)
            self.lin_dst = nn.Linear(in_dim[1], out_dim * num_heads, bias=False)

        # attention
        self.attn_src = nn.Parameter(torch.empty((1, num_heads, out_dim)))  # (1, H, C)
        self.attn_dst = nn.Parameter(torch.empty((1, num_heads, out_dim)))  # (1, H, C)

        # edge attention
        if self.edge_dim is not None:
            self.lin_edge = nn.Linear(edge_dim, out_dim * num_heads, bias=False)
            self.attn_edge = nn.Parameter(torch.empty(1, num_heads, out_dim))
        else:
            self.add_module('lin_edge', None)
            self.register_parameter('attn_edge', None)

        # concat heads
        concat_out_dim = out_dim * num_heads

        # residual
        if self.residual:
            self.lin_res = nn.Linear(
                in_dim if isinstance(in_dim, int) else in_dim[1],
                concat_out_dim if self.concat else out_dim,
                bias=False,
            )
        else:
            self.add_module('lin_res', None)

        # bias
        if bias:
            self.bias = nn.Parameter(torch.empty(concat_out_dim if self.concat else out_dim))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.lin is not None:
            init.xavier_uniform_(self.lin.weight)
        if self.lin_src is not None:
            init.xavier_uniform_(self.lin_src.weight)
        if self.lin_dst is not None:
            init.xavier_uniform_(self.lin_dst.weight)
        if self.lin_edge is not None:
            init.xavier_uniform_(self.lin_edge.weight)
        if self.lin_res is not None:
            init.xavier_uniform_(self.lin_res.weight)
        init.xavier_uniform_(self.attn_src)
        init.xavier_uniform_(self.attn_dst)
        if self.attn_edge is not None:
            init.xavier_uniform_(self.attn_edge)
        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(
        self,
        x: Union[Tensor, Tuple[Tensor, Tensor]],
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        dim_size: Optional[int] = None,
        return_attention_weights: Optional[bool] = None,
    ) -> Union[
        Tensor,
        Tuple[Tensor, Tensor, Tensor],
    ]:
        r"""Forward computation.

        Args:
            x (Tensor or Tuple[Tensor, Tensor]): Node feature matrix.
            edge_index (Tensor): Graph edge index tensor with shape
                :obj:`[2, num_edges]` or sparse adj tensor.
            edge_attr (Tensor, optional): Edge feature matrix with shape
                :obj:`[num_edges, edge_dim]`.
            dim_size (int, optional): The size of aggregator output, i.e. num of destination nodes.
                If None, infer from edge_index.
            return_attention_weights (bool, optional): If set to :obj:`True`, will additionally
                return the:obj: edge_index, alpha` containing the edge_index and the corresponding attention weights.

        Returns:
            Tensor or Tuple[Tensor, Tensor, Tensor]: The output feature matrix.
                - Tensor: The output feature matrix.
                - Tuple[Tensor, Tensor, Tensor]: If :obj:`return_attention_weights=True`, will additionally
                return the :obj:`edge_index, alpha` containing the edge_index and the corresponding attention weights.
        """
        # set `return_attention_weights`, tell message() to keep the attention weights
        self.return_attention_weights = return_attention_weights
        self.alpha = None

        # residual
        if isinstance(x, Tensor):
            resi = self.lin_res(x) if self.residual else None
        else:
            resi = self.lin_res(x[1]) if self.residual else None

        # linear transformation
        if isinstance(x, Tensor):
            assert self.lin is not None, "Bipartite GATConv requires a tuple x = (x_src, x_dst) input."
            x_src = x_dst = self.lin(x).view(-1, self.num_heads, self.out_dim)  # (N, H, C)
        else:
            x_src = self.lin_src(x[0]).view(-1, self.num_heads, self.out_dim)  # (N, H, C)
            x_dst = self.lin_dst(x[1]).view(-1, self.num_heads, self.out_dim)  # (M, H, C)

        x = (x_src, x_dst)

        # attention
        alpha_src = (x_src * self.attn_src).sum(dim=-1)  # (N, H)
        alpha_dst = (x_dst * self.attn_dst).sum(dim=-1)  # (M, H)
        alpha = (alpha_src, alpha_dst)

        # propagate (edge attn if edge_attr is not None) (default aggregator: sum)
        out = self.propagate(x, edge_index, alpha=alpha, edge_attr=edge_attr, dim_size=dim_size)  # (N, H, C)

        # concat or average
        if self.concat:
            out = out.view(-1, self.num_heads * self.out_dim)  # (N, H * C)
        else:
            out = out.mean(dim=self.head_dim)  # (N, C)

        # residual
        if resi is not None:
            out += resi

        # bias
        if self.bias is not None:
            out += self.bias

        # return attention weights or not
        if self.return_attention_weights:
            if edge_index.is_sparse:
                return out, set_values(edge_index, self.alpha), self.alpha
            else:
                return out, edge_index, self.alpha
        else:
            return out

    def message(
        self,
        x: Tuple[Tensor, Tensor],  # (N, H, C)
        edge_index: Tensor,
        alpha: Tuple[Tensor, Tensor],
        edge_attr: Optional[Tensor]
    ) -> Tensor:
        edge_index, _ = self.__unify_edgeindex__(edge_index)

        # calculate alpha
        alpha_i, alpha_j = self.retrieve_feats(alpha, edge_index)
        alpha = alpha_i + alpha_j  # (E, H)

        index = edge_index[0]
        if index.numel() == 0:
            return alpha

        if edge_attr is not None and self.lin_edge is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.unsqueeze(-1)
            edge_attr = self.lin_edge(edge_attr)
            edge_attr = edge_attr.view(-1, self.num_heads, self.out_dim)  # (E, H, C)
            alpha += (edge_attr * self.attn_edge).sum(dim=-1)  # (E, H)

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = F.softmax(alpha, dim=-1)
        # alpha = F.dropout(alpha, p=self.dropout, training=self.training)  # (E, H)

        # return_attention_weights
        if self.return_attention_weights:
            self.alpha = alpha

        # message
        x_j = self.retrieve_feats(x, edge_index, dim=1)
        return alpha.unsqueeze(-1) * x_j  # (E, H, C)

    def __repr__(self):
        return (f"{self.__class__.__name__}"
                f"({self.in_dim}, {self.out_dim})"
                f" num_heads={self.num_heads}, concat={self.concat}")
