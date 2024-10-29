from typing import Optional, Callable

import torch
import torch.nn.functional as F
from torch.nn import Parameter
from torch import Tensor


class SAGEConv(torch.nn.Module):
    r"""Simple SAGEConv layer, similar to <https://arxiv.org/abs/1706.02216>.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        aggr_methods (str): The aggregation method to use,
        *e.g.*, `mean`, `max_pooling`, `mean_pooling`, `gcn`, `lstm`.
        act: (Callable): The activation function is applied to aggreagtion,
            the default function is ReLU.
        concat (bool): If set to `False`, the multi-head attentions are
            averaged instead of concatenated.
        dropout (float): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. The default value is 0.0.
        bias (bool): If set to `False`, no bias terms are added into
            the final output.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 aggr_method: str = 'mean_pooling',
                 act: Optional[Callable] = torch.nn.ReLU(),
                 concat: bool = False,
                 dropout: float = 0.0,
                 bias: bool = False):
        self.in_channels = in_channels
        self.out_channels = out_channels

        if aggr_method == 'mean':
            self.aggr_module = MeanAggregator(
                in_channels, out_channels, act, dropout, dropout, bias
            )
        elif aggr_method == 'max_pooling':
            self.aggr_module = MaxPoolingAggregator(
                in_channels,
                in_channels,
                out_channels,
                act,
                concat,
                dropout,
                bias,
            )
        elif aggr_method == 'mean_pooling':
            self.aggr_module = MeanPoolingAggregator(
                in_channels,
                in_channels,
                out_channels,
                act, concat,
                dropout,
                bias
            )
        elif aggr_method == 'gcn':
            self.aggr_module = GCNAggregator(
                in_channels, out_channels, act, dropout, bias
            )
        elif aggr_method == 'lstm':
            self.aggr_module = LSTMAggregator(
                in_channels,
                in_channels,
                out_channels,
                act, concat,
                dropout,
                bias
            )
        else:
            raise NotImplementedError(
                f"Method of {aggr_method} is not implemented!"
            )

    def forward(self, self_vecs: Tensor, neigh_vecs: Tensor):
        return self.aggr_module(self_vecs, neigh_vecs)


class Aggregator(torch.nn.Module):
    r"""A base aggregate implementation."""
    def __init__(self,
                 self_channels: int,
                 neigh_channels: int,
                 out_channels: int,
                 act: Optional[Callable] = torch.nn.ReLU(),
                 concat: bool = False,
                 dropout: float = 0.0,
                 bias: bool = False):
        super().__init__()
        self.self_channels = self_channels
        self.neigh_channels = neigh_channels
        self.out_channels = out_channels
        self.act = act
        self.concat = concat
        self.dropout = dropout

        self.self_weight = Parameter(
            torch.empty(self_channels, out_channels)
        )
        self.neigh_weight = Parameter(
            torch.empty(neigh_channels, out_channels)
        )

        if bias and concat:
            self.bias = Parameter(torch.empty(2*out_channels))
        elif bias:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        # reset parameter
        torch.nn.init.xavier_normal_(self.self_weight)
        torch.nn.init.xavier_normal_(self.neigh_weight)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(self, self_vecs: Tensor, neigh_vecs: Tensor):
        self_vecs = F.dropout(self_vecs, self.dropout, self.training)
        neigh_vecs = F.dropout(neigh_vecs, self.dropout, self.training)

        # aggregate
        neigh_vecs = self.aggregate(self_vecs, neigh_vecs)

        from_self = torch.mm(self_vecs, self.self_weight)
        from_neigh = torch.mm(neigh_vecs, self.neigh_weight)

        if self.concat:
            out = torch.cat([from_self, from_neigh], dim=-1)
        else:
            out = from_self + from_neigh

        if self.bias is not None:
            out = out + self.bias

        return self.act(out)

    def aggregate(self, self_vecs: Tensor, neigh_vecs: Tensor):
        raise NotImplementedError

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.self_channels}, {self.neigh_channels}, '  # noqa
                f'{self.out_channels}')  # noqa


class MeanAggregator(Aggregator):
    r"""Aggregate neighbor features with mean operation."""
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 act: Optional[Callable] = torch.nn.ReLU(),
                 concat: bool = False,
                 dropout: float = 0.0,
                 bias: bool = False):
        super().__init__(
            self_channels=in_channels,
            neigh_channels=in_channels,
            out_channels=out_channels,
            act=act,
            concat=concat,
            dropout=dropout,
            bias=bias
        )

    def aggregate(self, self_vecs: Tensor, neigh_vecs: Tensor):
        return torch.mean(neigh_vecs, dim=1)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.self_channels}, '  # noqa
                f'{self.out_channels}')  # noqa


class MaxPoolingAggregator(Aggregator):
    r"""Aggregate neighbor features with max-pooling operation."""
    def __init__(self,
                 in_channels: int,
                 hidden_channels: int,
                 out_channels: int,
                 act: Optional[Callable] = torch.nn.ReLU(),
                 concat: bool = False,
                 dropout: float = 0.0,
                 bias: bool = False):
        super().__init__(
            self_channels=in_channels,
            neigh_channels=hidden_channels,
            out_channels=out_channels,
            act=act,
            concat=concat,
            dropout=dropout,
            bias=bias
        )

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout)
        )

    def aggregate(self, self_vecs: Tensor, neigh_vecs: Tensor):
        neigh_vecs = self.mlp(neigh_vecs)
        neigh_max = torch.max(neigh_vecs, dim=1)
        return neigh_max

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.self_channels}, {self.neigh_channels}, '  # noqa
                f'{self.out_channels}')  # noqa


class MeanPoolingAggregator(Aggregator):
    r"""Aggregate neighbor features with mean-pooling operation."""
    def __init__(self,
                 in_channels: int,
                 hidden_channels: int,
                 out_channels: int,
                 act: Optional[Callable] = torch.nn.ReLU(),
                 concat: bool = False,
                 dropout: float = 0.0,
                 bias: bool = False):
        super().__init__(
            self_channels=in_channels,
            neigh_channels=hidden_channels,
            out_channels=out_channels,
            act=act,
            concat=concat,
            dropout=dropout,
            bias=bias
        )

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout)
        )

    def aggregate(self, self_vecs: Tensor, neigh_vecs: Tensor):
        neigh_vecs = self.mlp(neigh_vecs)
        neigh_means = torch.mean(neigh_vecs, dim=1)
        return neigh_means

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.self_channels}, {self.neigh_channels}, '  # noqa
                f'{self.out_channels}')  # noqa


class GCNAggregator(torch.nn.Module):
    r"""Aggregate neighbor features with GCN operation."""
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 act: Optional[Callable] = torch.nn.ReLU(),
                 dropout: float = 0.0,
                 bias: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act = act
        self.dropout = dropout

        self.weight = Parameter(torch.empty(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.weight)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(self, self_vecs, neigh_vecs):
        self_vecs = F.dropout(self_vecs, self.dropout, self.training)
        neigh_vecs = F.dropout(neigh_vecs, self.dropout, self.training)

        means = torch.mean(torch.cat(
            [self_vecs.unsqueeze(1), neigh_vecs], dim=1), dim=1
        )
        out = torch.mm(means, self.weight)

        if self.bias is not None:
            out = out + self.bias

        return self.act(out)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}')


class LSTMAggregator(Aggregator):
    r"""Aggregate neighbor features with LSTM operation."""
    def __init__(self,
                 in_channels: int,
                 hidden_channels: int,
                 out_channels: int,
                 act: Optional[Callable] = torch.nn.ReLU(),
                 concat: bool = False,
                 dropout: float = 0.0,
                 bias: bool = False):
        super().__init__(
            self_channels=in_channels,
            neigh_channels=hidden_channels,
            out_channels=out_channels,
            act=act,
            concat=concat,
            dropout=dropout,
            bias=bias
        )

        self.rnn = torch.nn.LSTM(
            in_channels,
            hidden_channels,
            batch_first=True
        )

    def init_state(self, batch_size):
        hs = torch.zeros(1, batch_size, self.neigh_channels)
        cs = torch.zeros(1, batch_size, self.neigh_channels)
        return (hs, cs)

    def aggregate(self, self_vecs: Tensor, neigh_vecs: Tensor):
        N = neigh_vecs.shape[0]
        state = self.init_state(N)
        outputs, _ = self.rnn(neigh_vecs, state)
        return outputs

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.self_channels}, {self.neigh_channels}, '  # noqa
                f'{self.out_channels}')  # noqa
