from typing import Optional, Union

import torch
from torch import Tensor
from torch.nn import Linear, Parameter
from torch.sparse import Tensor as SparseTensor
import torch.nn.init as init

from rllm.nn.conv.graph_conv import MessagePassing


class GCNConv(MessagePassing):
    r"""The GCN (Graph Convolutional Network) model re-implementation with message passing,
    based on the `"Semi-supervised Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`__ paper.

    This model applies convolution operations to graph-structured data,
    allowing for the aggregation of feature information from neighboring nodes.

    .. math::
        \mathbf{X}^{\prime} = \mathbf{\hat{A}} \mathbf{X} \mathbf{W}

    Args:
        in_dim (int): Size of each input sample.
        out_dim (int): Size of each output sample.
        bias (bool): If set to `False`,
            no bias terms are added into the final output.

    Shapes:

        - **input:**

            node features :math:`(|\mathcal{V}|, F_{in})`

            edge_index is sparse adjacency matrix :math:`(|\mathcal{V}|, |\mathcal{V}|)`
            or edge list :math:`(2, |\mathcal{E}|)`

        - **output:**

            node features :math:`(|\mathcal{V}|, F_{out})`
    """

    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            bias: bool = True,
    ):
        super().__init__(aggr='gcn')
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear = Linear(in_dim, out_dim, bias=False)
        if bias:
            self.bias = Parameter(torch.empty(out_dim))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.xavier_normal_(self.linear.weight)
        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(
            self,
            x: Tensor,
            edge_index: Union[Tensor, SparseTensor],
            edge_weight: Optional[Tensor] = None,
            dim_size: Optional[int] = None,
    ) -> Tensor:
        x = self.linear(x)
        out = self.propagate(x, edge_index, edge_weight=edge_weight, dim_size=dim_size)
        if self.bias is not None:
            out += self.bias
        return out

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.in_dim}, " f"{self.out_dim})"
