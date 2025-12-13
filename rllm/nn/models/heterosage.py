from typing import List, Dict, Optional, Tuple

import torch
from torch import Tensor
from torch.nn import LayerNorm, ModuleDict, ModuleList

from rllm.nn.conv.graph_conv import SAGEConv


class HeteroSAGE(torch.nn.Module):
    r"""The herterogeneous version of the GraphSAGE model.

    Args:
        node_types (List[str]): The list of node types.
        edge_types (List[Tuple[str, str, str]]): The list of edge types.
        channels (int): The number of channels.
        aggr (str): The aggregation method.
        num_layers (int): The number of layers.
    """
    def __init__(
        self,
        node_types: List[str],
        edge_types: List[Tuple[str, str, str]],
        hidden_dim: int,
        aggr: str = "mean",
        num_layers: int = 2,
    ):
        super().__init__()

        self.edge_type_mapping = {
            edge_type: "__".join(edge_type) for edge_type in edge_types
        }

        self.convs = ModuleList()
        for _ in range(num_layers):
            conv_dict = ModuleDict()
            for edge_type in edge_types:
                conv_dict[self.edge_type_mapping[edge_type]] = \
                    SAGEConv(hidden_dim, hidden_dim, aggr=aggr)
            self.convs.append(conv_dict)

        self.norms = torch.nn.ModuleList()
        for _ in range(num_layers):
            norm_dict = torch.nn.ModuleDict()
            for node_type in node_types:
                norm_dict[node_type] = LayerNorm(hidden_dim)
            self.norms.append(norm_dict)

        self.reset_parameters()

    def reset_parameters(self):
        for conv_dict in self.convs:
            for conv in conv_dict.values():
                conv.reset_parameters()
        for norm_dict in self.norms:
            for norm in norm_dict.values():
                norm.reset_parameters()

    def forward(
        self,
        x_dict: Dict[str, Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], Tensor]
    ) -> Dict[str, Tensor]:
        for layer in range(len(self.convs)):
            conv_dict = self.convs[layer]
            # apply graph conv to each edge type and
            # aggregate along the edge type
            dst_dict = {}
            for edge_type, edge_index in edge_index_dict.items():
                src, _, dst = edge_type
                x_src = x_dict[src]
                x_dst = x_dict[dst]
                if dst not in dst_dict:
                    dst_dict[dst] = []
                dst_dict[dst].append(
                    conv_dict[self.edge_type_mapping[edge_type]]((x_src, x_dst), edge_index)
                )
            for dst, x_list in dst_dict.items():
                x_stack = torch.stack(x_list, dim=0)
                # update x_dict
                x_dict[dst] = torch.sum(x_stack, dim=0, keepdim=False)
            # apply layer norm to each node type
            for node_type, x in x_dict.items():
                x_dict[node_type] = self.norms[layer][node_type](x)
            # apply activation function
            x_dict = {key: x.relu() for key, x in x_dict.items()}

        return x_dict
