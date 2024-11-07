from typing import Optional, Callable

import torch
import torch.nn.functional as F
from torch.nn import Parameter
from torch import Tensor


class SAGEConv(torch.nn.Module):
    r"""Simple SAGEConv layer, similar to <https://arxiv.org/abs/1706.02216>.

    Args:
        in_dim (int): Size of each input sample.
        out_dim (int): Size of each output sample.
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

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        aggr_method: str = "mean_pooling",
        act: Optional[Callable] = torch.nn.ReLU(),
        concat: bool = False,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        self.in_dim = in_dim
        self.out_dim = out_dim

        if aggr_method == "mean":
            self.aggr_module = MeanAggregator(
                in_dim, out_dim, act, dropout, dropout, bias
            )
        elif aggr_method == "max_pooling":
            self.aggr_module = MaxPoolingAggregator(
                in_dim, in_dim, out_dim, act, concat, dropout, bias
            )
        elif aggr_method == "mean_pooling":
            self.aggr_module = MeanPoolingAggregator(
                in_dim, in_dim, out_dim, act, concat, dropout, bias
            )
        elif aggr_method == "gcn":
            self.aggr_module = GCNAggregator(in_dim, out_dim, act, dropout, bias)
        elif aggr_method == "lstm":
            self.aggr_module = LSTMAggregator(
                in_dim, in_dim, out_dim, act, concat, dropout, bias
            )
        else:
            raise NotImplementedError(f"Method of {aggr_method} is not implemented!")

    def forward(self, self_vecs: Tensor, neigh_vecs: Tensor):
        return self.aggr_module(self_vecs, neigh_vecs)


class Aggregator(torch.nn.Module):
    r"""A base aggregate implementation."""

    def __init__(
        self,
        self_dim: int,
        neigh_dim: int,
        out_dim: int,
        act: Optional[Callable] = torch.nn.ReLU(),
        concat: bool = False,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        self.self_dim = self_dim
        self.neigh_dim = neigh_dim
        self.out_dim = out_dim
        self.act = act
        self.concat = concat
        self.dropout = dropout

        self.self_weight = Parameter(torch.empty(self_dim, out_dim))
        self.neigh_weight = Parameter(torch.empty(neigh_dim, out_dim))

        if bias and concat:
            self.bias = Parameter(torch.empty(2 * out_dim))
        elif bias:
            self.bias = Parameter(torch.empty(out_dim))
        else:
            self.register_parameter("bias", None)

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
        return (
            f"{self.__class__.__name__}({self.self_dim}, {self.neigh_dim}, "  # noqa
            f"{self.out_dim}"
        )  # noqa


class MeanAggregator(Aggregator):
    r"""Aggregate neighbor features with mean operation."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        act: Optional[Callable] = torch.nn.ReLU(),
        concat: bool = False,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__(
            self_dim=in_dim,
            neigh_dim=in_dim,
            out_dim=out_dim,
            act=act,
            concat=concat,
            dropout=dropout,
            bias=bias,
        )

    def aggregate(self, self_vecs: Tensor, neigh_vecs: Tensor):
        return torch.mean(neigh_vecs, dim=1)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.self_dim}, "  # noqa
            f"{self.out_dim}"
        )  # noqa


class MaxPoolingAggregator(Aggregator):
    r"""Aggregate neighbor features with max-pooling operation."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        act: Optional[Callable] = torch.nn.ReLU(),
        concat: bool = False,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__(
            self_dim=in_dim,
            neigh_dim=hidden_dim,
            out_dim=out_dim,
            act=act,
            concat=concat,
            dropout=dropout,
            bias=bias,
        )

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
        )

    def aggregate(self, self_vecs: Tensor, neigh_vecs: Tensor):
        neigh_vecs = self.mlp(neigh_vecs)
        neigh_max = torch.max(neigh_vecs, dim=1)
        return neigh_max

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.self_dim}, {self.neigh_dim}, "  # noqa
            f"{self.out_dim}"
        )  # noqa


class MeanPoolingAggregator(Aggregator):
    r"""Aggregate neighbor features with mean-pooling operation."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        act: Optional[Callable] = torch.nn.ReLU(),
        concat: bool = False,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__(
            self_dim=in_dim,
            neigh_dim=hidden_dim,
            out_dim=out_dim,
            act=act,
            concat=concat,
            dropout=dropout,
            bias=bias,
        )

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
        )

    def aggregate(self, self_vecs: Tensor, neigh_vecs: Tensor):
        neigh_vecs = self.mlp(neigh_vecs)
        neigh_means = torch.mean(neigh_vecs, dim=1)
        return neigh_means

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.self_dim}, {self.neigh_dim}, "  # noqa
            f"{self.out_dim}"
        )  # noqa


class GCNAggregator(torch.nn.Module):
    r"""Aggregate neighbor features with GCN operation."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        act: Optional[Callable] = torch.nn.ReLU(),
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.act = act
        self.dropout = dropout

        self.weight = Parameter(torch.empty(in_dim, out_dim))

        if bias:
            self.bias = Parameter(torch.empty(out_dim))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.weight)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(self, self_vecs, neigh_vecs):
        self_vecs = F.dropout(self_vecs, self.dropout, self.training)
        neigh_vecs = F.dropout(neigh_vecs, self.dropout, self.training)

        means = torch.mean(
            torch.cat([self_vecs.unsqueeze(1), neigh_vecs], dim=1), dim=1
        )
        out = torch.mm(means, self.weight)

        if self.bias is not None:
            out = out + self.bias

        return self.act(out)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.in_dim}, " f"{self.out_dim}"


class LSTMAggregator(Aggregator):
    r"""Aggregate neighbor features with LSTM operation."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        act: Optional[Callable] = torch.nn.ReLU(),
        concat: bool = False,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__(
            self_dim=in_dim,
            neigh_dim=hidden_dim,
            out_dim=out_dim,
            act=act,
            concat=concat,
            dropout=dropout,
            bias=bias,
        )

        self.rnn = torch.nn.LSTM(in_dim, hidden_dim, batch_first=True)

    def init_state(self, batch_size):
        hs = torch.zeros(1, batch_size, self.neigh_dim)
        cs = torch.zeros(1, batch_size, self.neigh_dim)
        return (hs, cs)

    def aggregate(self, self_vecs: Tensor, neigh_vecs: Tensor):
        N = neigh_vecs.shape[0]
        state = self.init_state(N)
        outputs, _ = self.rnn(neigh_vecs, state)
        return outputs

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.self_dim}, {self.neigh_dim}, "  # noqa
            f"{self.out_dim}"
        )  # noqa
