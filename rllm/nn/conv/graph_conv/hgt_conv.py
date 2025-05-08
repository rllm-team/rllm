from typing import Union, Tuple, List, Dict

import math
import torch
from torch import Tensor
from torch.sparse import Tensor as SparseTensor

from rllm.utils import seg_softmax
from rllm.nn.conv.graph_conv import MessagePassing


class HGTConv(MessagePassing):
    r"""The Heterogeneous Graph Transformer (HGT) layer
    implementation with message passing,
    as introduced in the `"Heterogeneous Graph Transformer"
    <https://arxiv.org/abs/2003.01332>`__ paper.

    Args:
        in_dim (Union[int, Dict[str, int]]):
            Size of each input sample of every node type.
        out_dim (int): Size of each output sample of every node type.
        metadata (Tuple[List[str], List[Tuple[str, str]]]): The metadata of
            the heterogeneous graph, *i.e.* its node and edge types given
            by a list of strings and a list of string triplets, respectively.
        num_heads (int, optional):
            Number of multi-head attentions (default: :obj:`1`).
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training (default: :obj:`0.0`).
        use_pre_encoder (bool, optional):
            Whether to use pre-encoder (default: :obj:`False`).
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
        dropout: float = 0.0,
        use_pre_encoder: bool = False,
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

        # params init
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads

        # node feats pre_encoder
        self.lin_dict = None
        if use_pre_encoder:
            self.lin_dict = torch.nn.ModuleDict()
            for node_type, in_dim in self.in_dim.items():
                self.lin_dict[node_type] = torch.nn.Linear(in_dim, out_dim, bias=True)
                self.in_dim[node_type] = out_dim

        # multi-head node attention
        self.q_lin = torch.nn.ModuleDict()
        self.k_lin = torch.nn.ModuleDict()
        self.v_lin = torch.nn.ModuleDict()
        self.a_lin = torch.nn.ModuleDict()
        self.skip = torch.nn.ParameterDict()
        self.dropout = torch.nn.Dropout(dropout)

        # Initialize parameters for each node type
        for node_type, in_dim in self.in_dim.items():
            self.q_lin[node_type] = torch.nn.Linear(in_dim, out_dim, bias=True)
            self.k_lin[node_type] = torch.nn.Linear(in_dim, out_dim, bias=True)
            self.v_lin[node_type] = torch.nn.Linear(in_dim, out_dim, bias=True)
            self.a_lin[node_type] = torch.nn.Linear(in_dim, out_dim, bias=True)
            self.skip[node_type] = torch.nn.Parameter(torch.tensor(1.0))

        # meta-relation attention
        self.a_rel = torch.nn.ParameterDict()
        self.m_rel = torch.nn.ParameterDict()
        self.p_rel = torch.nn.ParameterDict()
        hidden_dim = out_dim // num_heads

        # Initialize parameters for each edge type
        for edge_type in edge_types:
            edge_type = "__".join(edge_type)

            # a_rel weights
            a_weight = torch.nn.Parameter(
                torch.empty((num_heads, hidden_dim, hidden_dim), requires_grad=True)
            )
            torch.nn.init.trunc_normal_(a_weight)
            self.a_rel[edge_type + "a"] = a_weight

            # m_rel weights
            m_weight = torch.nn.Parameter(
                torch.empty((num_heads, hidden_dim, hidden_dim), requires_grad=True)
            )
            torch.nn.init.trunc_normal_(m_weight)
            self.m_rel[edge_type + "m"] = m_weight

            # p_rel weights
            self.p_rel[edge_type] = torch.nn.Parameter(torch.ones(num_heads))

    def forward(
        self,
        x_dict: Dict[str, Tensor],
        edge_index_dict: Dict[Tuple[str, str], Union[Tensor, SparseTensor]],
    ):
        H, D = self.num_heads, self.out_dim // self.num_heads
        k_dict, q_dict, v_dict, out_node_dict, out_dict = {}, {}, {}, {}, {}

        # Linear projection, q, k, v
        for node_type, x in x_dict.items():
            if self.lin_dict is not None:
                x = self.lin_dict[node_type](x)

            out_node_dict[node_type] = x
            k_dict[node_type] = self.k_lin[node_type](x).view(-1, H, D)  # (N, F_in) -> (N, H, D)
            q_dict[node_type] = self.q_lin[node_type](x).view(-1, H, D)
            v_dict[node_type] = self.v_lin[node_type](x).view(-1, H, D)

            out_dict[node_type] = []

        # Iterate over edge_types
        for edge_type, edge_index in edge_index_dict.items():
            src_node_type, dst_node_type = edge_type
            edge_type = "__".join(edge_type)

            # multi-head node attention
            # (H, N, D) @ (H, D, D) -> (H, N, D) -> (N, H, D)
            a_rel = self.a_rel[edge_type + "a"]
            k = (k_dict[src_node_type].transpose(1, 0) @ a_rel).transpose(1, 0)  # (N, H, D)

            # (H, N, D) @ (H, D, D) -> (H, N, D) -> (N, H, D)
            m_rel = self.m_rel[edge_type + "m"]
            v = (v_dict[src_node_type].transpose(1, 0) @ m_rel).transpose(1, 0)  # (N, H, D)

            # meta-relation attention
            edge_index, _ = self.__unify_edgeindex__(edge_index)
            src_index, dst_index = edge_index

            # q, k, v
            q_dst = torch.index_select(q_dict[dst_node_type], 0, dst_index)  # (E, H, D)
            k_src = torch.index_select(k, 0, src_index)  # (E, H, D)
            v_src = torch.index_select(v, 0, src_index)  # (E, H, D)

            rel = self.p_rel[edge_type]  # (H)

            out = self.propagate(
                _,
                edge_index=edge_index,
                q_dst=q_dst,
                k_src=k_src,
                v_src=v_src,
                rel=rel,
                dim_size=x_dict[dst_node_type].size(0)
            )

            out_dict[dst_node_type].append(out)

        # out
        for node_type, outs in out_dict.items():
            # node type aggregation
            outs = torch.stack(outs)  # (k, N, out_dim)
            out = torch.sum(outs, dim=0, keepdim=False)  # (N, out_dim)

            # FFN
            out = self.a_lin[node_type](out)
            alpha = torch.sigmoid(self.skip[node_type])

            # skip connection
            out = alpha * out + (1 - alpha) * out_node_dict[node_type]
            out_dict[node_type] = out

        # out_dict: (Num_node_type, N, out_dim)
        return out_dict

    def message_and_aggregate(self, edge_index, q_dst, k_src, v_src, rel, dim_size):
        # alpha: (E, H)
        alpha = (k_src * q_dst).sum(dim=-1) * rel
        alpha = alpha / math.sqrt(q_dst.size(-1))
        alpha = self.dropout(seg_softmax(alpha, edge_index[1], edge_index[1].max().item() + 1))
        # out: (E, H, D)
        out = v_src * alpha.unsqueeze(-1)
        out = out.view(-1, self.out_dim)  # (E, out_dim[H*D])

        # aggregate -> (N, out_dim)
        return self.aggr_module(out, edge_index[1, :], dim=self.node_dim, dim_size=dim_size)

    def __repr__(self):
        return (f'{self.__class__.__name__}({self.out_dim}, ',
                f'num_heads={self.num_heads})')
