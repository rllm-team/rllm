from typing import Union, Tuple, List, Dict

import torch
from torch import Tensor
from torch.nn import Parameter
import torch.nn.functional as F

from rllm.nn.conv.gat_conv import GATConv


class HANConv(torch.nn.Module):
    r"""The Heterogeneous Graph Attention Network (HAN) model,
    as introduced in the `"Heterogeneous Graph Attention
    Network" <https://arxiv.org/abs/1903.07293>`__ paper.

    This model leverages the power of attention mechanisms in the context of
    heterogeneous graphs, allowing for the learning of node representations
    that capture both the structure and the multifaceted nature of the graph.

    Args:
        in_channels (int or Dict[str, int]): Size of each input sample of every
            node type.
        metadata (Tuple[List[str], List[Tuple[str, str, str]]]): The metadata
            of the heterogeneous graph, *i.e.* its node and edge types given
            by a list of strings and a list of string triplets, respectively.
        heads (int, optional):
            Number of multi-head-attentions, the default values is 1.
        negative_slop (float):
            LeakyReLU angle of the negative slope, the default value is 0.2.
        dropout (float): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. The default value is 0.
    """
    def __init__(self,
                 in_channels: Union[int, Dict[str, int]],
                 out_channels: int,
                 metadata: Tuple[List[str], List[Tuple[str, str]]],
                 heads: int = 1,
                 negative_slope: float = 0.2,
                 dropout: float = 0.0):
        super().__init__()
        if not isinstance(in_channels, dict):
            in_channels = {node_type: in_channels for node_type in metadata[0]}

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout

        hid_dim = out_channels // heads
        self.fc_dict = torch.nn.ModuleDict()
        for node_type, in_dim in self.in_channels.items():
            self.fc_dict[node_type] = torch.nn.Linear(in_dim, hid_dim)

        self.conv_dict = torch.nn.ModuleDict()
        for edge_type in metadata[1]:
            edge_type = '__'.join(edge_type)
            self.conv_dict[edge_type] = GATConv(
                in_channels=(hid_dim, hid_dim),
                out_channels=hid_dim,
                heads=heads,
                negative_slope=negative_slope,
                dropout=dropout
            )

        self.q = Parameter(torch.empty(1, out_channels))
        self.fc_k = torch.nn.Linear(out_channels, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.q)
        self.fc_k.reset_parameters()
        # TODO: init dict parameter.

    def forward(self,
                x_dict: Dict[str, Tensor],
                adj_dict: Dict[Tuple[str, str], Tensor],
                return_semantic_att_weights: bool = False):
        # Embedding node features into semantic space.
        out_node_dict, out_dict = {}, {}
        for node_type, x in x_dict.items():
            out_node_dict[node_type] = self.fc_dict[node_type](x)
            out_dict[node_type] = []

        # For different edges,
        # we use corresponding GAT to capture graph information.
        for edge_type, adj in adj_dict.items():
            src_type = edge_type[0]
            tgt_type = edge_type[1]

            conv_inputs = (out_node_dict[src_type], out_node_dict[tgt_type])
            out = self.conv_dict['__'.join(edge_type)](conv_inputs, adj)
            out_dict[tgt_type].append(out)

        # For different GAT convolution outputs,
        # we use attention to fuse features.
        semantic_attn_dict = {}
        for node_type, outs in out_dict.items():
            out, score = self.attention(outs)
            out_dict[node_type] = out
            semantic_attn_dict[node_type] = score

        if return_semantic_att_weights:
            return out_dict, semantic_attn_dict

        return out_dict

    def attention(self, xs: Tensor):
        # [num_edge_types, num_nodes, out_channels*heads]
        xs = torch.stack(xs, dim=0)
        # [num_edge_types, out_channels*heads]
        key = torch.tanh(self.fc_k(xs)).mean(1)
        # [num_edge_types]
        attn_score = (self.q * key).sum(-1)
        score = F.softmax(attn_score, dim=0)
        out = (xs * score[:, None, None]).sum(0)
        return out, score
