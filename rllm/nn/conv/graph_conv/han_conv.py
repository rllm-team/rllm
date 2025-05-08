from typing import Union, Tuple, List, Dict

import torch
from torch import Tensor
from torch.sparse import Tensor as SparseTensor
from torch.nn import Parameter
import torch.nn.functional as F

from rllm.utils import seg_softmax
from rllm.nn.conv.graph_conv import MessagePassing


class HANConv(MessagePassing):
    r"""The Heterogeneous Graph Attention Network (HAN) model
    implementation with message passing,
    as introduced in the `"Heterogeneous Graph Attention
    Network" <https://arxiv.org/abs/1903.07293>`__ paper.

    This model leverages the power of attention mechanisms in the context of
    heterogeneous graphs, allowing for the learning of node representations
    that capture both the structure and the multifaceted nature of the graph.

    Args:
        in_dim (int or Dict[str, int]): Size of each input sample of every
            node type.
        out_dim (int): Size of each output sample of every node type.
        metadata (Tuple[List[str], List[Tuple[str, str, str]]]): The metadata
            of the heterogeneous graph, *i.e.* its node and edge types given
            by a list of strings and a list of string triplets, respectively.
        num_heads (int, optional):
            Number of multi-head-attentions (default: :obj:`1`).
        negative_slop (float):
            LeakyReLU angle of the negative slope (default: :obj:`0.2`).
        dropout (float): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training (default: :obj:`0`).
        aggr (str): The aggregation method to use.
            Defaults: 'sum'.
        **kwargs (optional): Additional arguments of :class:`MessagePassing`.
    """

    node_dim = 0

    def __init__(
        self,
        in_dim: Union[int, Dict[str, int]],
        out_dim: int,
        metadata: Tuple[List[str], List[Tuple[str, str]]],
        num_heads: int = 1,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        *,
        aggr: str = "sum",
        **kwargs
    ):
        # default use 'sum' aggregator
        super().__init__(aggr=aggr, aggr_kwargs=kwargs)

        node_types, edge_types = metadata

        # If in_dim is not dict, use the same in_dim for all node types
        if not isinstance(in_dim, dict):
            in_dim = {node_type: in_dim for node_type in node_types}

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.negative_slope = negative_slope
        self.dropout = dropout

        # Linear projection
        self.lin_dict = torch.nn.ModuleDict()
        for node_type, in_dim in self.in_dim.items():
            self.lin_dict[node_type] = torch.nn.Linear(in_dim, out_dim)

        # Multi-head node attention
        self.lin_src = torch.nn.ParameterDict()
        self.lin_dst = torch.nn.ParameterDict()
        hidden_dim = out_dim // num_heads
        for edge_type in edge_types:
            edge_type = "__".join(edge_type)
            self.lin_src[edge_type] = torch.nn.Parameter(torch.empty(1, num_heads, hidden_dim))
            self.lin_dst[edge_type] = torch.nn.Parameter(torch.empty(1, num_heads, hidden_dim))

        # meta-path attention
        self.k_lin = torch.nn.Linear(out_dim, out_dim, bias=True)
        self.q = Parameter(torch.empty(1, out_dim))

        # reset parameters
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.k_lin.weight)
        torch.nn.init.xavier_normal_(self.q)
        for edge_type in self.lin_src.keys():
            torch.nn.init.xavier_normal_(self.lin_src[edge_type])
            torch.nn.init.xavier_normal_(self.lin_dst[edge_type])

    def forward(
        self,
        x_dict: Dict[str, Tensor],
        edge_index_dict: Dict[Tuple[str, str], Union[Tensor, SparseTensor]],
        return_semantic_attn_weights: bool = False,
    ):
        H, D = self.num_heads, self.out_dim // self.num_heads
        node_dict, out_dict = {}, {}

        # Linear projection
        for node_type, x in x_dict.items():
            node_dict[node_type] = self.lin_dict[node_type](x).view(-1, H, D)  # (N, in_dim) -> (N, H, D)
            out_dict[node_type] = []

        # Iterate over edge types
        for edge_type, edge_index in edge_index_dict.items():
            src_node_type, dst_node_type = edge_type
            edge_type = "__".join(edge_type)

            # multi-head node attention
            # (N, H, D) * (1, H, D) -> (N, H)
            src_x = node_dict[src_node_type]
            dst_x = node_dict[dst_node_type]
            alpha_src = (self.lin_src[edge_type] * src_x).sum(dim=-1)
            alpha_dst = (self.lin_dst[edge_type] * dst_x).sum(dim=-1)
            alpha = (alpha_src, alpha_dst)

            # message passing
            out = self.propagate(
                None,
                edge_index=edge_index,
                src_x=src_x,
                alpha=alpha,
                dim_size=dst_x.size(0),
            )

            out = F.relu(out)
            out_dict[dst_node_type].append(out)

        # meta-path attention
        semantic_attn_dict = {}
        for node_type, outs in out_dict.items():
            outs = torch.stack(outs, dim=0)  # (num_edge_types, N, out_dim)
            k = torch.tanh(self.k_lin(outs)).mean(dim=1, keepdim=False)  # (num_edge_types, out_dim)
            attn_score = (self.q * k).sum(dim=-1, keepdim=False)  # (num_edge_types)
            attn = F.softmax(attn_score, dim=0)
            outs = attn.view(-1, 1, 1) * outs
            out = outs.sum(dim=0, keepdim=False)
            out_dict[node_type] = out
            semantic_attn_dict[node_type] = attn

        if return_semantic_attn_weights:
            return out_dict, semantic_attn_dict

        return out_dict

    def message_and_aggregate(self, edge_index, src_x, alpha, dim_size):
        edge_index, _ = self.__unify_edgeindex__(edge_index)
        alpha_src, alpha_dst = self.retrieve_feats(alpha, edge_index)
        src_x = self.retrieve_feats(src_x, edge_index, dim=0)  # (E, H, D)

        # alpha: (E, H)
        alpha = alpha_src + alpha_dst
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = seg_softmax(alpha, edge_index[1, :], num_segs=dim_size)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # msg: (E, out_dim[H*D])
        msgs = src_x * alpha.unsqueeze(-1)  # (E, H, D) * (E, H) -> (E, H, D)
        msgs = msgs.view(-1, self.out_dim)  # (E, H, D) -> (E, H*D)
        return self.aggr_module(msgs, edge_index[1, :], dim=self.node_dim, dim_size=dim_size)

    def __repr__(self):
        return (f'{self.__class__.__name__}({self.out_dim}, ',
                f'num_heads={self.num_heads})')
