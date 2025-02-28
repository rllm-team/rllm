from typing import Optional, Union

import torch
from torch import Tensor
from torch.nn import Linear, Parameter
import torch.nn.init as init
from torch.sparse import Tensor as SparseTensor

from rllm.nn.conv.graph_conv.message_passing import MessagePassing


class GCNConv(MessagePassing):
    r"""
    """

    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            bias: bool = True,
    ):
        super().__init__()
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
            num_nodes: Optional[int] = None
    ) -> Tensor:
        x = self.linear(x)
        out = self.propagate(x, edge_index, edge_weight=edge_weight, num_nodes=num_nodes)
        if self.bias is not None:
            out += self.bias
        return out

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.in_dim}, " f"{self.out_dim})"
