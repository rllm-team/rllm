from typing import Optional, Callable, Union, Tuple, Dict

import torch
from torch import Tensor
from torch.nn import Linear
from torch.sparse import Tensor as SparseTensor
import torch.nn.functional as F

from .transformer_conv import GTransformerConv
from .sage_conv import SAGEConv


class RelGNNConv(GTransformerConv):
    r"""The convolution layer of RelGNN model
    from `Relational Graph Neural Networks with Composite Message Passing
    <https://arxiv.org/abs/2306.14803>`_ paper.

    Args:
        attn_type (str): The attention type.
        in_dim (Tuple[int, int]): The input dimension of source node
            and destination node.
        out_dim (int): The output dimension.
        num_heads (int): The number of attention heads.
        aggr (str): The aggregation method.
        simplified_MP (bool): Whether to use simplified message passing.
        bias (bool): Whether to use bias.
    """
    def __init__(
        self,
        attn_type: str,
        in_dim: int,
        out_dim: int,
        num_heads: int,
        aggr: str,
        simplified_MP=False,
        bias=True,
        **kwargs,
    ):
        super().__init__(
            in_dim=(in_dim, in_dim),
            out_dim=out_dim,
            num_heads=num_heads,
            bias=bias,
            **kwargs,
        )

        self.attn_type = attn_type
        if attn_type == 'dim-fact-dim':
            self.aggr_conv = SAGEConv(in_dim, out_dim, aggr=aggr)
        else:
            self.aggr_conv = None
        self.simplified_MP = simplified_MP
        self.final_proj = Linear(num_heads * out_dim, out_dim, bias=bias)

    def reset_parameters(self):
        self.final_proj.reset_parameters()
        if self.attn_type == 'dim-fact-dim':
            self.aggr_conv.reset_parameters()
        return super().reset_parameters()

    def forward(
        self,
        x,
        edge_index,
        edge_weight = None,
        return_attention_weights = False,
    ):
        # dim-dim
        if self.attn_type == 'dim-dim':
            if self.simplified_MP and edge_index.shape[1] == 0:
                return None
            out = super().forward(
                x,
                edge_index,
                edge_weight,
                return_attention_weights
            )
            return self.final_proj(out)

        # dim-fact-dim
        edge_attn, edge_aggr = edge_index

        src_aggr, dst_aggr, dst_attn = x

        if self.simplified_MP:
            if edge_attn.shape[1] == 0:
                return None

            if edge_aggr.shape[1] == 0:
                src_attn = dst_aggr
            else:
                src_attn = self.aggr_conv((src_aggr, dst_aggr), edge_aggr)
        else:
            src_attn = self.aggr_conv((src_aggr, dst_aggr), edge_aggr)

        out = super().forward(
            (src_attn, dst_attn),
            edge_attn,
            edge_weight,
            return_attention_weights
        )

        return self.final_proj(out), src_attn