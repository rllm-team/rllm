from typing import Optional, Union

from torch import Tensor
from torch.sparse import Tensor as SparseTensor

from rllm.nn.conv.graph_conv import MessagePassing


class LazyConv(MessagePassing):
    r"""The Lazy Graph Convolution re-implementation with message passing,

    .. math::
        \mathbf{\hat{A}} = \mathbf{\tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}}}

        \mathbf{X}^{(k+1)} = \left[\beta\mathbf{\hat{A}} + (1-\beta)\mathbf{I}\right] \mathbf{X}^{(k)}
    """
    node_dim: int = 0

    def __init__(self, beta: float = 0.5):
        super().__init__(aggr='sum')
        self.beta = beta

    def forward(
            self,
            x: Tensor,
            edge_index: Union[Tensor, SparseTensor],
            edge_weight: Optional[Tensor] = None,
    ) -> Tensor:
        x = self.propagate(x, edge_index, edge_weight=edge_weight, dim=self.node_dim)
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
        return f"{self.__class__.__name__}(beta: {self.beta})"
