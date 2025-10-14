from typing import Union, Tuple

import torch
from torch import Tensor
from torch.sparse import Tensor as SparseTensor
import torch.nn.init as init

from rllm.utils import set_values, seg_softmax
from rllm.nn.conv.graph_conv import MessagePassing


class GATConv(MessagePassing):
    r"""The GAT (Graph Attention Network) model
    implementation with message passing,
    based on the `"Graph Attention Networks"
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
        in_dim (int): Size of each input sample.
        out_dim (int): Size of each output sample.
        num_heads (int): Number of multi-head-attentions, the default value is 1.
        concat (bool): If set to `False`, the multi-head attentions
            are averaged instead of concatenated.
        negative_slope (float): LeakyReLU angle of the negative slope,
            the default value is 0.2.
        dropout (float): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. The default value is 0.
        bias (bool):
            If set to `False`, no bias terms are added into the final output.
        skip_connection (bool): If set to :obj:`True`, the layer will add
            a learnable skip-connection. (default: :obj:`False`)

    Shapes:

        - **input:**

            node features :math:`(|\mathcal{V}|, F_{in})`

            edge_index is sparse adjacency matrix :math:`(|\mathcal{V}|, |\mathcal{V}|)`
            or edge list :math:`(2, |\mathcal{E}|)`

        - **output:**

            node features :math:`(|\mathcal{V}|, F_{out})`
    """
    node_dim = 0
    head_dim = 1

    def __init__(
        self,
        in_dim: Union[int, Tuple[int, int]],
        out_dim: int,
        num_heads: int = 8,
        concat: bool = False,
        negative_slope: float = 0.2,
        dropout: float = 0.6,
        bias: bool = True,
        skip_connection: bool = False,
        **kwargs
    ):
        super().__init__(aggr='add', **kwargs)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.concat = concat
        self.skip_connection = skip_connection

        if isinstance(in_dim, int):
            self.lin = torch.nn.Linear(in_dim, out_dim * num_heads, bias=False)
        else:
            in_dim = tuple(in_dim)
            self.lin_src = torch.nn.Linear(in_dim[0], out_dim * num_heads, bias=False)
            self.lin_dst = torch.nn.Linear(in_dim[1], out_dim * num_heads, bias=False)

        if self.skip_connection:
            self.lin_skip = torch.nn.Linear(
                in_features=in_dim[1] if isinstance(in_dim, tuple) else in_dim,
                out_features=out_dim * num_heads if self.concat else out_dim,
                bias=False
            )

        # attention weights
        self.attn_src = torch.nn.Parameter(torch.Tensor(1, num_heads, out_dim))
        self.attn_dst = torch.nn.Parameter(torch.Tensor(1, num_heads, out_dim))

        if bias and concat:
            self.bias = torch.nn.Parameter(torch.empty(num_heads * out_dim))
        elif bias and not concat:
            self.bias = torch.nn.Parameter(torch.empty(out_dim))
        else:
            self.register_parameter('bias', None)

        self.leaky_relu = torch.nn.LeakyReLU(negative_slope)
        self.dropout = torch.nn.Dropout(p=dropout)

        self.reset_parameters()

    def reset_parameters(self):
        if self.bias is not None:
            init.zeros_(self.bias)
        init.xavier_normal_(self.attn_src)
        init.xavier_normal_(self.attn_dst)

    def forward(
        self,
        x: Union[Tensor, Tuple[Tensor, Tensor]],
        edge_index: Union[Tensor, SparseTensor],
        return_attention_weights: bool = False,
    ) -> Union[
        Tensor,
        Tuple[Tensor, Tuple[Tensor, Tensor]],
        Tuple[Tensor, Tuple[SparseTensor, Tensor]],
    ]:
        self.return_attention_weights = return_attention_weights
        # Linear projection
        if isinstance(x, Tensor):
            if self.skip_connection:
                skip_res = self.lin_skip(x)
            x = self.lin(x).view(-1, self.num_heads, self.out_dim)  # (N, H, D)
            x_src = x_dst = x
        else:
            if self.skip_connection:
                skip_res = self.lin_skip(x[1])
            x_src = self.lin_src(x[0]).view(-1, self.num_heads, self.out_dim)  # (N_src, H, D)
            x_dst = self.lin_dst(x[1]).view(-1, self.num_heads, self.out_dim)  # (N_dst, H, D)

        num_nodes = x_dst.size(0)  # N_dst

        # node attention
        alpha_src = (x_src * self.attn_src).sum(dim=-1)  # (N_src, H)
        alpha_dst = (x_dst * self.attn_dst).sum(dim=-1)  # (N_dst, H)

        # out: (N_dst, H, D)
        out = self.propagate(
            None,
            edge_index=edge_index,
            alpha=(alpha_src, alpha_dst),
            x_src=x_src,
            dim_size=num_nodes,
        )

        if self.concat:
            out = out.view(-1, self.num_heads * self.out_dim)  # (N_dst, H * D)
        else:
            out = out.mean(dim=self.head_dim, keepdim=False)  # (N_dst, D)

        if self.skip_connection:
            out = out + skip_res

        if self.bias is not None:
            out += self.bias

        if return_attention_weights:
            if edge_index.is_sparse:
                return out, (set_values(edge_index, self.attn_weights), self.attn_weights)
            else:
                return out, (edge_index, self.attn_weights)
        else:
            return out

    def message_and_aggregate(self, edge_index, alpha, x_src, dim_size):
        edge_index, _ = self.__unify_edgeindex__(edge_index)
        x_src = self.retrieve_feats(x_src, edge_index, dim=0, retrieve_dim=self.node_dim)

        # attention scores
        # alpha_src / alpha_dst: (E, H)
        alpha_src, alpha_dst = self.retrieve_feats(alpha, edge_index, retrieve_dim=self.node_dim)
        # alpha: (E, H)
        alpha = self.leaky_relu(alpha_src + alpha_dst)
        alpha = seg_softmax(alpha, edge_index[1], num_segs=dim_size)
        alpha = self.dropout(alpha)

        self.attn_weights = alpha.clone().detach() if self.return_attention_weights else None

        # msgs: (E, H, D)
        msgs = x_src * alpha.unsqueeze(-1)
        return self.aggr_module(msgs, edge_index[1], dim=self.node_dim, dim_size=dim_size)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.in_dim}, "
            f"{self.out_dim}, num_heads={self.num_heads})"
        )
