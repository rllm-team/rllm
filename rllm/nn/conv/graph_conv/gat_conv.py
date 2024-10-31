from typing import Union, Tuple
from torch import Tensor
import torch
import torch.nn as nn


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
        in_dim (int): Size of each input sample.
        out_dim (int): Size of each output sample.
        heads (int): Number of multi-head-attentions, the default value is 1.
        concat (bool): If set to `False`, the multi-head attentions
            are averaged instead of concatenated.
        negative_slop (float): LeakyReLU angle of the negative slope,
            the default value is 0.2.
        skip_connection (bool): If set to :obj:`True`, the layer will add
            a learnable skip-connection. (default: :obj:`False`)
        dropout (float): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. The default value is 0.
        bias (bool):
            If set to `False`, no bias terms are added into the final output.

    Shapes:
        - **input:**

            node features : (N, F_IN),

            sparse adjacency matrix : (N, N),

        - **output:**

            node features : (N, F_OUT)
    """

    nodes_dim = 0  # node dimension/axis
    head_dim = 1  # attention head dimension/axis

    def __init__(
        self,
        in_dim: Union[int, Tuple[int, int]],
        out_dim: int,
        heads: int = 8,
        concat: bool = True,
        negative_slope: float = 0.2,
        skip_connection: bool = False,
        dropout: float = 0.6,
        bias: bool = True,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.concat = concat
        self.skip_connection = skip_connection

        if isinstance(in_dim, int):
            self.lin_src = self.lin_tgt = torch.nn.Linear(
                in_dim, heads * out_dim, bias=False
            )
        else:
            self.lin_src = torch.nn.Linear(in_dim[0], heads * out_dim, bias=False)
            self.lin_tgt = torch.nn.Linear(in_dim[1], heads * out_dim, bias=False)

        if self.skip_connection:
            self.skip = nn.Linear(in_dim[1], heads * out_dim, bias=False)

        # Define the attention source/target weights as a learnable parameter.
        # Shape: (1, heads, out_dim), where:
        # - 1 represents a batch dimension
        # - heads is the number of attention heads
        # - out_dim is the size of each head's output feature
        self.attention_target = nn.Parameter(torch.Tensor(1, heads, out_dim))
        self.attention_source = nn.Parameter(torch.Tensor(1, heads, out_dim))
        if bias and concat:
            self.bias = nn.Parameter(torch.empty(heads * out_dim))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.empty(out_dim))
        else:
            self.register_parameter("bias", None)

        self.leakyReLU = nn.LeakyReLU(negative_slope)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(p=dropout)

        self.reset_parameters()

    def reset_parameters(self):
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)
        nn.init.xavier_uniform_(self.attention_target)
        nn.init.xavier_uniform_(self.attention_source)

    def forward(self, inputs: Union[Tensor, Tuple[Tensor, Tensor]], adj: Tensor):

        if isinstance(inputs, Tensor):
            inputs = (inputs, inputs)

        # The node features are linearly varied to obtain the features of
        # the number of attention heads * the number of hidden units.
        # shape = (N, F_IN) -> (N, H, F_OUT)
        num_nodes = inputs[1].size(0)
        nodes_features_src = self.lin_src(inputs[0]).view(-1, self.heads, self.out_dim)
        nodes_features_tgt = self.lin_tgt(inputs[1]).view(-1, self.heads, self.out_dim)

        # shape = (N, H, F_OUT) * (1, H, F_OUT) -> (N, H, F_OUT) -> (N, H)
        scores_source = (nodes_features_src * self.attention_source).sum(dim=-1)
        scores_target = (nodes_features_tgt * self.attention_target).sum(dim=-1)

        # Gets the source and target of each edge
        idx_source = adj.coalesce().indices()[0]
        idx_target = adj.coalesce().indices()[1]

        # scores shape = (N, H) -> (E, H)
        # nodes_features_selected shape = (E, H, F_OUT)
        # E - number of edges in the graph
        scores_source_selected = scores_source.index_select(self.nodes_dim, idx_source)
        scores_target_selected = scores_target.index_select(self.nodes_dim, idx_target)
        nodes_features_selected = nodes_features_src.index_select(
            self.nodes_dim, idx_source
        )

        # leakyReLU + softmax
        scores_edge = self.leakyReLU(scores_source_selected + scores_target_selected)
        scores_edge_exp = self.softmax(scores_edge)

        # The score for each side is calculated according to the neighbor.
        scores_edge_neigborhood = self.score_edge_wiht_neighborhood(
            scores_edge_exp, idx_target, num_nodes
        )

        # 1e-16 is theoretically not needed but is only there for numerical
        # stability (avoid div by 0) - due to the possibility of
        # the computer rounding a very small number all the way to 0.
        attentions_edge = scores_edge_exp / (scores_edge_neigborhood + 1e-16)
        attentions_edge = attentions_edge.unsqueeze(-1)
        attentions_edge = nn.functional.dropout(
            attentions_edge, p=0.6, training=self.training
        )

        # Element-wise product. Operator * does the same thing as torch.mul
        # shape = (E, H, F_OUT) * (E, H, 1) -> (E, H, F_OUT)
        # 1 gets broadcast into F_OUT
        nodes_features_weighted = nodes_features_selected * attentions_edge

        # aggregate neighborhood
        nodes_features_aggregated = self.aggregate_neighborhoods(
            nodes_features_weighted, idx_target, num_nodes
        )

        # Residual/skip connections, concat and bias
        out_nodes_features = self.skip_concat(inputs, nodes_features_aggregated)
        return out_nodes_features

    def aggregate_neighborhoods(self, nodes_features, idx_target, num_nodes):
        size = list(nodes_features.shape)
        size[self.nodes_dim] = num_nodes  # shape = (N, H, FOUT)
        nodes_features_aggregated = torch.zeros(
            size, dtype=nodes_features.dtype, device=nodes_features.device
        )
        idx_target_bd = self.expand_dim(idx_target, nodes_features)
        nodes_features_aggregated = nodes_features_aggregated.scatter_add(
            self.nodes_dim, idx_target_bd, nodes_features
        )
        return nodes_features_aggregated

    def skip_concat(self, in_dim, out_dim):
        if not out_dim.is_contiguous():
            out_dim = out_dim.contiguous()

        if self.concat:
            out_dim = out_dim.view(-1, self.heads * self.out_dim)
        else:
            out_dim = out_dim.mean(dim=self.head_dim)

        if self.skip_connection:
            if out_dim.shape[-1] == in_dim.shape[-1]:
                out_dim += in_dim.unsqueeze(1)
            else:
                out_dim += self.skip(in_dim).view(-1, self.heads * self.out_dim)

        if self.bias is not None:
            out_dim += self.bias

        return out_dim

    def score_edge_wiht_neighborhood(self, scores_edge_exp, idx_target, num_nodes):
        # The shape must be the same as in scores_edge_exp (required by scatter_add_)
        # i.e. from E -> (E, H)
        idx_target_broadcasted = self.expand_dim(idx_target, scores_edge_exp)

        # shape = (N, H), where N is the number of nodes
        # H the number of attention heads
        size = list(
            scores_edge_exp.shape
        )  # convert to list otherwise assignment is not possible
        size[self.nodes_dim] = num_nodes
        neighborhood_sums = torch.zeros(
            size, dtype=scores_edge_exp.dtype, device=scores_edge_exp.device
        )
        neighborhood_sums.scatter_add_(
            self.nodes_dim, idx_target_broadcasted, scores_edge_exp
        )

        # shape = (N, H) -> (E, H)
        return neighborhood_sums.index_select(self.nodes_dim, idx_target)

    def expand_dim(self, src, trg):
        for _ in range(src.dim(), trg.dim()):
            src = src.unsqueeze(-1)
        return src.expand_as(trg)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.in_dim}, "
            f"{self.out_dim}, heads={self.heads})"
        )
