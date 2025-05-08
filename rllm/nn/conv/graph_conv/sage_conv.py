from typing import Optional, Callable, Union, Tuple

import torch
from torch import Tensor
from torch.nn import Parameter
from torch.sparse import Tensor as SparseTensor
import torch.nn.functional as F

from rllm.nn.conv.graph_conv import MessagePassing
from rllm.nn.conv.graph_conv.aggrs import Aggregator


class SAGEConv(MessagePassing):
    r"""Simple SAGEConv layer implementation with message passing,
    as introduced in the
    `"Inductive Representation Learning on Large Graphs"
    <https://arxiv.org/abs/1706.02216>`__ paper.

    Supported aggregators:
        sum, mean, max_pool, mean_pool, gcn, lstm

    Args:
        in_dim (int): Size of each input sample.
        out_dim (int): Size of each output sample.
        aggr (str or Aggregator): The aggregation method to use,
            *e.g.*, `sum`, `mean`, `max_pool`, `mean_pool`, `gcn`, `lstm`.
        activation: (Callable): The activationivation function is applied to aggreagtion,
            the default function is ReLU.
        concat (bool): If set to `False`, the multi-head attentions are
            averaged instead of concatenated.
        dropout (float): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. The default value is 0.0.
        bias (bool): If set to `False`, no bias terms are added into
            the final output.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        aggr: Optional[Union[str, Aggregator]] = 'sum',
        activation: Optional[Callable] = F.relu,
        dropout: float = 0.0,
        bias: bool = False,
        **kwargs
    ):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.aggr = aggr
        self.activation = activation
        self.dropout = dropout

        if aggr == 'lstm':
            kwargs.setdefault('aggr_kwargs', {})
            kwargs['aggr_kwargs'].setdefault('in_dim', in_dim)
            kwargs['aggr_kwargs'].setdefault('out_dim', in_dim)
        elif aggr[-4:] == 'pool':
            kwargs.setdefault('aggr_kwargs', {})
            kwargs['aggr_kwargs'].setdefault('in_dim', in_dim)
            kwargs['aggr_kwargs'].setdefault('out_dim', in_dim)

        super().__init__(aggr=self.aggr, **kwargs)

        self.lin_neigh = torch.nn.Linear(in_dim, in_dim, bias=False)

        if aggr == 'gcn':
            self.register_module('self_lin', None)
        else:
            self.self_lin = torch.nn.Linear(in_dim, out_dim, bias=False)

        self.lin = torch.nn.Linear(in_dim, out_dim, bias=False)

        if bias:
            self.bias = Parameter(torch.empty(out_dim), requires_grad=True)
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.lin_neigh.weight)
        torch.nn.init.xavier_normal_(self.lin.weight)
        if self.self_lin is not None:
            torch.nn.init.xavier_normal_(self.self_lin.weight)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(
        self,
        x: Union[Tensor, Tuple[Tensor]],
        edge_index: Union[Tensor, SparseTensor],
        edge_weight: Optional[Tensor] = None,
    ):
        if isinstance(x, Tensor):
            x = [x, x]
        else:
            x = list(x)

        x[0] = F.dropout(x[0], p=self.dropout, training=self.training)
        x[1] = F.dropout(x[1], p=self.dropout, training=self.training)

        if self.aggr[-4:] != 'pool':
            x[0] = self.lin_neigh(x[0])  # (N, in_dim)

        if self.aggr == 'gcn' and self.self_lin is None:
            """GCN aggregator.
            Assuming the edge_index has been GCN normalized while preprocessing.
            """
            out = self.propagate(
                x[0],
                edge_index,
                edge_weight=edge_weight,
                dim_size=x[1].size(0)
            )
        elif self.aggr[-4:] == 'pool':
            out = self.propagate(x[0], edge_index, dim_size=x[1].size(0))
            out = self.lin_neigh(out)
        else:
            out = self.propagate(x[0], edge_index, dim_size=x[1].size(0))

        out = self.lin(out)  # (N, out_dim)

        if self.self_lin is not None:
            out += self.self_lin(x[1])  # (N, out_dim)

        if self.bias is not None:
            out += self.bias

        return self.activation(out) if self.activation else out

    def __repr__(self):
        return f"{self.__class__.__name__}({self.in_dim}, {self.out_dim})"
