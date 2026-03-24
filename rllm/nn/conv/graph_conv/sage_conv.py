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
            *e.g.*, :obj:`"sum"`, :obj:`"mean"`, :obj:`"max_pool"`,
            :obj:`"mean_pool"`, :obj:`"gcn"`, :obj:`"lstm"`.
            (default: :obj:`"sum"`)
        activation (Callable): The activation function applied after
            aggregation. (default: :obj:`F.relu`)
        dropout (float): Dropout probability applied to node features before
            aggregation. (default: :obj:`0.0`)
        bias (bool): If set to :obj:`False`, no bias terms are added into
            the final output. (default: :obj:`False`)
        dst_in_dim (Optional[int]): The input dimension of the destination
            nodes. If :obj:`None`, :obj:`in_dim` is used. Useful for
            heterogeneous graphs where source and destination nodes have
            different dimensions. (default: :obj:`None`)
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        aggr: Optional[Union[str, Aggregator]] = "sum",
        activation: Optional[Callable] = F.relu,
        dropout: float = 0.0,
        bias: bool = False,
        dst_in_dim: Optional[int] = None,
        **kwargs,
    ):
        aggr_name = aggr.lower() if isinstance(aggr, str) else None
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.aggr = aggr
        self.activation = activation
        self.dropout = dropout

        if aggr_name == "lstm":
            kwargs.setdefault("aggr_kwargs", {})
            kwargs["aggr_kwargs"].setdefault("in_dim", in_dim)
            kwargs["aggr_kwargs"].setdefault("out_dim", in_dim)
        elif aggr_name is not None and aggr_name.endswith("pool"):
            kwargs.setdefault("aggr_kwargs", {})
            kwargs["aggr_kwargs"].setdefault("in_dim", in_dim)
            kwargs["aggr_kwargs"].setdefault("out_dim", in_dim)

        super().__init__(aggr=self.aggr, **kwargs)

        self.lin_neigh = torch.nn.Linear(in_dim, in_dim, bias=False)

        if aggr == "gcn":
            self.register_module("self_lin", None)
        else:
            if dst_in_dim is not None:
                self.self_lin = torch.nn.Linear(dst_in_dim, out_dim, bias=False)
            else:
                self.self_lin = torch.nn.Linear(in_dim, out_dim, bias=False)

        self.lin = torch.nn.Linear(in_dim, out_dim, bias=False)

        if bias:
            self.bias = Parameter(torch.empty(out_dim), requires_grad=True)
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
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
        r"""Aggregate neighbor information and combine with destination features.

        Args:
            x (Union[Tensor, Tuple[Tensor]]):
                - Tensor input features for homogeneous graphs.
                - Tuple containing source and destination node features.
            edge_index (Union[Tensor, SparseTensor]): Graph connectivity in edge-list
                or sparse adjacency format.
            edge_weight (Optional[Tensor]): Optional edge weights used by certain
                aggregators such as ``gcn``.

        Returns:
            Tensor: Output embeddings for destination nodes.

        Example:
            >>> import torch
            >>> from rllm.nn.conv.graph_conv import SAGEConv
            >>> conv = SAGEConv(16, 8, aggr='sum')
            >>> x = torch.randn(4, 16)
            >>> edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
            >>> out = conv(x, edge_index)
            >>> out.shape
            torch.Size([4, 8])
        """
        if isinstance(x, Tensor):
            x = [x, x]
        else:
            x = list(x)

        x[0] = F.dropout(x[0], p=self.dropout, training=self.training)
        x[1] = F.dropout(x[1], p=self.dropout, training=self.training)

        aggr_name = self.aggr.lower() if isinstance(self.aggr, str) else ""
        if not aggr_name.endswith("pool"):
            x[0] = self.lin_neigh(x[0])  # (N, in_dim)

        if aggr_name == "gcn" and self.self_lin is None:
            """GCN aggregator.
            Assuming the edge_index has been GCN normalized while preprocessing.
            """
            out = self.propagate(
                x[0], edge_index, edge_weight=edge_weight, dim_size=x[1].size(0)
            )
        elif aggr_name.endswith("pool"):
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
