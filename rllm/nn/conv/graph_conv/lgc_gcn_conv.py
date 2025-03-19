from typing import Optional, Union

from torch import Tensor
from torch.sparse import Tensor as SparseTensor

from rllm.nn.conv.graph_conv import LGCConv


class GCNConv(LGCConv):
    r"""The GCN (Graph Convolutional Network) model re-implementation with
    LGCConv (Lazy Graph Convolution), based on the `"Semi-supervised Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`__ paper. and the `"From Cluster Assumption to Graph Convolution:
     Graph-based Semi-Supervised Learning Revisited" <https://arxiv.org/abs/2309.13599>`__ paper.

    While the LGConv model degenerates to the GCN model with :math:`\beta = 1`, the GCNConv
    can easily re-implemented in this way.

    Args:
        in_dim (int): Size of each input sample.
        out_dim (int): Size of each output sample.
        bias (bool): If set to `False`,
            no bias terms are added into the final output.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool = True,
    ):
        self.in_dim = in_dim
        self.out_dim = out_dim
        super().__init__(beta=1, with_param=True, in_dim=in_dim, out_dim=out_dim, bias=bias)

    def forward(
            self,
            x: Tensor,
            edge_index: Union[Tensor, SparseTensor],
            edge_weight: Optional[Tensor] = None,
    ) -> Tensor:
        return super().forward(x, edge_index, edge_weight=edge_weight)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.in_dim}, " f"{self.out_dim})"
