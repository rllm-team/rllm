import torch
from torch.nn import Parameter
from torch import Tensor


class GCNConv(torch.nn.Module):
    r"""The GCN (Graph Convolutional Network) model, based on the
    `"Semi-supervised Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`__ paper.

    This model applies convolution operations to graph-structured data,
    allowing for the aggregation of feature information from neighboring nodes.

    .. math::
        \mathbf{X}^{\prime} = \mathbf{\hat{A}} \mathbf{X} \mathbf{W}

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        bias (bool): If set to `False`,
            no bias terms are added into the final output.

    Shapes:

        - **input:**

            node features :math:`(|\mathcal{V}|, F_{in})`

            sparse adjacency matrix :math:`(|\mathcal{V}|, |\mathcal{V}|)`

        - **output:**

            node features :math:`(|\mathcal{V}|, F_{out})`
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 bias: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = Parameter(torch.empty(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.xavier_normal_(self.weight)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(self, inputs: Tensor, adj: Tensor):
        support = torch.mm(inputs, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels})')
