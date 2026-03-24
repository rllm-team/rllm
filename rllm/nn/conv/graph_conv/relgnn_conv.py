from torch.nn import Linear

from .transformer_conv import GTransformerConv
from .sage_conv import SAGEConv


class RelGNNConv(GTransformerConv):
    r"""The convolution layer of RelGNN model
    from `Relational Graph Neural Networks with Composite Message Passing
    <https://arxiv.org/abs/2306.14803>`_ paper.

    This layer supports both direct node-to-node attention and a factorized
    path that aggregates through latent factors before final attention.

    Args:
        attn_type (str): The attention type, one of :obj:`"dim-dim"` or
            :obj:`"dim-fact-dim"`.
        in_dim (int): The input feature dimension.
        out_dim (int): The output feature dimension per attention head.
        num_heads (int): The number of attention heads.
        aggr (str): The aggregation method for the factorized path.
        simplified_MP (bool): Whether to skip propagation when there are no
            edges. (default: :obj:`False`)
        bias (bool): Whether to add a bias term. (default: :obj:`True`)
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
        if attn_type == "dim-fact-dim":
            self.aggr_conv = SAGEConv(in_dim, out_dim, aggr=aggr)
        else:
            self.aggr_conv = None
        self.simplified_MP = simplified_MP
        self.final_proj = Linear(num_heads * out_dim, out_dim, bias=bias)

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.final_proj.reset_parameters()
        if self.attn_type == "dim-fact-dim":
            self.aggr_conv.reset_parameters()
        return super().reset_parameters()

    def forward(
        self,
        x,
        edge_index,
        edge_weight=None,
        return_attention_weights=False,
    ):
        r"""Run RelGNN attention with optional factorized message passing.

        Args:
            x: Node features. For ``dim-dim`` attention this is a Tensor.
                For ``dim-fact-dim`` attention this is a tuple
                ``(src_aggr, dst_aggr, dst_attn)``.
            edge_index: Graph connectivity. For ``dim-dim`` attention this is
                a ``[2, num_edges]`` tensor. For ``dim-fact-dim`` attention this
                is a tuple ``(edge_attn, edge_aggr)``.
            edge_weight: Optional edge features passed to transformer attention.
            return_attention_weights (bool): Whether to return attention scores
                from the transformer attention path. (default: :obj:`False`)

        Returns:
            For ``dim-dim`` attention: Tensor output embeddings, or None when
            simplified message passing skips empty-edge propagation.
            For ``dim-fact-dim`` attention: Tuple of output embeddings and
            intermediate source features, or None when skipped.

        Example:
            >>> import torch
            >>> from rllm.nn.conv.graph_conv import RelGNNConv
            >>> conv = RelGNNConv('dim-dim', 8, 4, num_heads=2, aggr='sum')
            >>> x = torch.randn(4, 8)
            >>> edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
            >>> out = conv(x, edge_index)
            >>> out.shape
            torch.Size([4, 4])
        """
        # dim-dim
        if self.attn_type == "dim-dim":
            if self.simplified_MP and edge_index.shape[1] == 0:
                return None
            out = super().forward(x, edge_index, edge_weight, return_attention_weights)
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
            (src_attn, dst_attn), edge_attn, edge_weight, return_attention_weights
        )

        return self.final_proj(out), src_attn
