from typing import Union, Tuple

import torch
import torch.nn.functional as F
from torch.nn import Parameter
from torch import Tensor

from rllm.utils.sparse import get_indices


class GATConv(torch.nn.Module):
    r"""The GAT (Graph Attention Network) model, based on the
    `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`__ paper.

    In particular, this implementation utilizes sparse attention mechanisms
    to handle graph-structured data,
    similiar to <https://github.com/Diego999/pyGAT>.

    .. math::
        \mathbf{x}^{\prime}_i = \sum_{j \in \mathcal{N}(i) \cup \{ i \}}
        \alpha_{i,j}\mathbf{\Theta}_t\mathbf{x}_{j}

    where the attention coefficients :math:`\alpha_{i,j}` are computed as:

    .. math::
        \alpha_{i,j} =\frac{\exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        \mathbf{\Theta} \mathbf{x}_i+ \mathbf{a}^{\top} \mathbf{
        \Theta}\mathbf{x}_j\right)\right)}
        {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
        \exp\left(\mathrm{LeakyReLU}\left(
        \mathbf{a}^{\top}  \mathbf{\Theta} \mathbf{x}_i
        + \mathbf{a}^{\top}\mathbf{\Theta}\mathbf{x}_k
        \right)\right)}.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        heads (int): Number of multi-head-attentions, the default value is 1.
        concat (bool): If set to `False`, the multi-head attentions
            are averaged instead of concatenated.
        negative_slop (float): LeakyReLU angle of the negative slope,
            the default value is 0.2.
        dropout (float): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. The default value is 0.
        bias (bool):
            If set to `False`, no bias terms are added into the final output.

    Shapes:
        - **input:**

            node features :math:`(|\mathcal{V}|, F_{in})`,

            sparse adjacency matrix :math:`(|\mathcal{V}|, |\mathcal{V}|)`,

        - **output:**

            node features :math:`(|\mathcal{V}|, F_{out}\times N_{heads})`
    """

    def __init__(self,
                 in_channels: Union[int, Tuple[int, int]],
                 out_channels: int,
                 heads: int = 1,
                 concat: bool = True,
                 negative_slope: float = 0.2,
                 dropout: float = 0.0,
                 bias: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat

        self.dropout = dropout

        if isinstance(in_channels, int):
            self.fc_src = self.fc_tgt = torch.nn.Linear(
                in_channels, heads * out_channels, bias=False
            )
        else:
            self.fc_src = torch.nn.Linear(
                in_channels[0], heads * out_channels, bias=False
            )
            self.fc_tgt = torch.nn.Linear(
                in_channels[1], heads * out_channels, bias=False
            )

        self.attention = Parameter(torch.empty(1, heads, 2 * out_channels))

        if bias and concat:
            self.bias = Parameter(torch.empty(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self.leakyrelu = torch.nn.LeakyReLU(negative_slope)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.attention)

        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(
        self,
        inputs: Union[Tensor, Tuple[Tensor, Tensor]],
        adj: Tensor
    ):
        import torch_sparse
        if isinstance(inputs, Tensor):
            inputs = (inputs, inputs)

        N, M = adj.shape
        device = inputs[0].device
        h_src = self.fc_src(inputs[0]).view(-1, self.heads, self.out_channels)
        h_tgt = self.fc_tgt(inputs[1]).view(-1, self.heads, self.out_channels)

        edge = get_indices(adj)

        edge_h = torch.cat(
            (h_src[edge[0, :], :, :], h_tgt[edge[1, :], :, :]),
            dim=2
        )
        edge_e = torch.exp(-self.leakyrelu((edge_h * self.attention).sum(-1)))
        assert not torch.isnan(edge_e).any()
        edge_e = F.dropout(edge_e, self.dropout, self.training)

        hs = []
        edge_t = edge[[1, 0]]
        for i in range(self.heads):
            h_prime = torch_sparse.spmm(
                edge_t, edge_e[:, i], M, N, h_src[:, i, :]
            )
            assert not torch.isnan(edge_e).any()
            e_rowsum = torch_sparse.spmm(
                edge_t, edge_e[:, i], M, N,
                torch.ones(size=(N, 1), device=device)
            )
            h_prime = h_prime.div(e_rowsum.clamp(1e-10))
            hs.append(h_prime)
        out = torch.stack(hs, dim=1)

        if self.concat:
            out = out.view(-1, self.heads*self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias
        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')
