import math
import typing
from typing import Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Linear
import torch.nn.functional as F

from rllm.utils import seg_softmax
from .message_passing import MessagePassing


class GTransformerConv(MessagePassing):
    r"""The graph transformer operator from the `"Masked Label Prediction:
    Unified Message Passing Model for Semi-Supervised Classification"
    <https://arxiv.org/abs/2009.03509>`_ paper.

    Args:
        in_dim (Tuple[int, int]): Size of each input sample
            (for source and target nodes).
        out_dim (int): Size of each output sample.
        num_heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`).
        concat (bool, optional): If set to `False`, the multi-head attentions
            are averaged instead of concatenated. (default: :obj:`True`)
        beta (bool, optional): If set to :obj:`True`, the layer will add
            a learnable skip-connection with learnable weight.
            (default: :obj:`False`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`).
        edge_dim (int, optional): Size of each edge feature.
            (default: :obj:`None`, which means no edge features are used).
        bias (bool, optional): If set to `False`, no bias terms are added into
            the final output. (default: :obj:`True`).
        root_weight (bool, optional): If set to :obj:`False`, the layer will
            not use the root node feature for message update.
            (default: :obj:`True`)
    """

    _alpha: Optional[Tensor]
    _node_dim = 0

    def __init__(
        self,
        in_dim: Tuple[int, int],
        out_dim: int,
        num_heads: int = 1,
        concat: bool = True,
        beta: bool = False,
        dropout: float = 0.,
        edge_dim: Optional[int] = None,
        bias: bool = True,
        root_weight: bool = True,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = num_heads
        self.beta = beta and root_weight
        self.root_weight = root_weight
        self.concat = concat
        self.dropout = dropout
        self.edge_dim = edge_dim
        self._alpha = None

        if isinstance(in_dim, int):
            in_dim = (in_dim, in_dim)

        self.lin_key = Linear(in_dim[0], num_heads * out_dim)
        self.lin_query = Linear(in_dim[1], num_heads * out_dim)
        self.lin_value = Linear(in_dim[0], num_heads * out_dim)
        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, num_heads * out_dim, bias=False)
        else:
            self.lin_edge = self.register_parameter('lin_edge', None)

        if concat:
            self.lin_skip = Linear(in_dim[1], num_heads * out_dim,
                                   bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * num_heads * out_dim, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)
        else:
            self.lin_skip = Linear(in_dim[1], out_dim, bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * out_dim, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)

    def reset_parameters(self):
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        if self.lin_edge is not None:
            self.lin_edge.reset_parameters()
        self.lin_skip.reset_parameters()
        if self.beta:
            self.lin_beta.reset_parameters()

    def forward(
        self,
        x: Union[Tensor, Tuple[Tensor, Tensor]],
        edge_index: Tensor,
        edge_weight: Optional[Tensor] = None,
        return_attention_weights: bool = False,
    ):
        r"""Supports edge_index only."""
        H, C = self.heads, self.out_dim

        if isinstance(x, Tensor):
            x = (x, x)

        query = self.lin_query(x[1]).view(-1, H, C)
        key = self.lin_key(x[0]).view(-1, H, C)
        value = self.lin_value(x[0]).view(-1, H, C)

        out = self.propagate(
            None,
            edge_index,
            query=query,
            key=key,
            value=value,
            dim_size=x[1].size(0),  # num of dst nodes
            edge_weight=edge_weight,
        )

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_dim)
        else:
            out = out.mean(dim=1)

        if self.root_weight:
            x_r = self.lin_skip(x[1])
            if self.lin_beta is not None:
                beta = self.lin_beta(torch.cat([out, x_r, out - x_r], dim=-1))
                beta = beta.sigmoid()
                out = beta * x_r + (1 - beta) * out
            else:
                out = out + x_r

        if return_attention_weights:
            assert alpha is not None
            return out, (edge_index, alpha)
        else:
            return out

    def message_and_aggregate(
        self,
        edge_index: Tensor,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        dim_size: int,
        edge_weight: Optional[Tensor],
    ) -> Tensor:

        query_i = query.index_select(0, edge_index[1])
        key_j = key.index_select(0, edge_index[0])
        value_j = value.index_select(0, edge_index[0])

        edge_attr = None
        if self.lin_edge is not None:
            assert edge_weight is not None
            edge_attr = self.lin_edge(edge_weight).view(-1, self.heads,
                                                      self.out_dim)
            key_j = key_j + edge_attr

        alpha = (query_i * key_j).sum(dim=-1) / math.sqrt(self.out_dim)
        alpha = seg_softmax(alpha, edge_index[1], num_segs=dim_size)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = value_j
        if edge_attr is not None:
            out = out + edge_attr

        out = out * alpha.view(-1, self.heads, 1)
        return self.aggr_module(
            out,
            edge_index[1],
            dim=self._node_dim,
            dim_size=dim_size
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.in_dim}, "
            f"{self.out_dim}, num_heads={self.heads})"
        )