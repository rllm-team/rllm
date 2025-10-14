from typing import Optional, Union

import torch
from torch import Tensor
from torch.sparse import Tensor as SparseTensor

from rllm.nn.conv.graph_conv import MessagePassing


class LGCConv(MessagePassing):
    r"""The LGC (Lazy Graph Convolution) implementation with message passing,
    based on the `"From Cluster Assumption to Graph Convolution:
    Graph-based Semi-Supervised Learning Revisited"
    <https://arxiv.org/abs/2309.13599>`__ paper.

    This model use hyperparameter :math:`\beta` to control the message attribution of
    both neighbor nodes and the node itself:

        - If :math:`\beta = 1`, the model is equivalent to the graph convolution form of GCN model.
            (if `with_param` is `True`, the model is equivalent to the GCN model)

        - If :math:`\beta = 0`, the model only focus on the node itself.

    .. math::
        \mathbf{\hat{A}} = \mathbf{\tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}}}

        \mathbf{X}^{(k+1)} = \left[\beta\mathbf{\hat{A}} + (1-\beta)\mathbf{I}\right] \mathbf{X}^{(k)}

    Args:
        beta (float): The hyperparameter :math:`\beta` to control the message attribution.
        with_param (bool): If set to `True`, the model will learn a linear transformation
            for node features.
        in_dim (int): Size of each input sample.
        out_dim (int): Size of each output sample.
        bias (bool): If set to `False`, no bias terms are added into the final output.
            Only available when `with_param` is `True`.

    Shapes:

        - **input:**

            node features :math:`(|\mathcal{V}|, F_{in})`

            edge_index is sparse adjacency matrix :math:`(|\mathcal{V}|, |\mathcal{V}|)`
            or edge list :math:`(2, |\mathcal{E}|)`

        - **output:**

            node features :math:`(|\mathcal{V}|, F_{out})`
    """

    node_dim: int = 0

    def __init__(
        self,
        beta: float = 0.5,
        *,
        with_param: bool = False,
        in_dim: Optional[int] = None,
        out_dim: Optional[int] = None,
        bias: bool = False,
    ):
        super().__init__(aggr='sum')
        self.beta = beta
        self.with_param = with_param

        if self.with_param:
            assert in_dim is not None and out_dim is not None, "in_dim and out_dim should be provided"
            self.lin = torch.nn.Linear(in_dim, out_dim, bias=False)
            if bias:
                self.bias = torch.nn.Parameter(torch.empty(out_dim))
            else:
                self.register_parameter("bias", None)

            self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.xavier_normal_(self.lin.weight)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(
            self,
            x: Tensor,
            edge_index: Union[Tensor, SparseTensor],
            edge_weight: Optional[Tensor] = None,
    ) -> Tensor:
        if self.with_param:
            x = self.lin(x)

        x = self.propagate(x, edge_index, edge_weight=edge_weight, dim=self.node_dim)

        if self.with_param and self.bias is not None:
            x = x + self.bias
        return x

    def message_and_aggregate(self, x, edge_index, edge_weight, dim):
        r"""Lazy Graph Convolution

        .. math::
            lazy_msgs = \beta * gcn_msgs + (1-\beta) * \mathbf{X}
        """

        edge_index, ew = self.__unify_edgeindex__(edge_index)
        if edge_weight is None and ew is not None:
            edge_weight = ew

        src_index = edge_index[0, :]
        gcn_msgs = x.index_select(dim=0, index=src_index)
        if edge_weight is not None:
            gcn_msgs = gcn_msgs * edge_weight.view(-1, 1)
        gcn_msgs = self.aggr_module(gcn_msgs, edge_index[1, :].squeeze(), dim=dim, dim_size=x.size(0))
        return self.beta * gcn_msgs + (1 - self.beta) * x

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(beta: {self.beta}, with_param: {self.with_param})"
