from typing import Any, Dict, List, Tuple, Optional
from collections import defaultdict
import warnings

import torch
from torch import Tensor
from torch.nn import ModuleDict, ModuleList

from rllm.types import ColType, StatType
from rllm.data import HeteroGraphData
from rllm.nn.models import TableResNet
from rllm.nn.conv.graph_conv.relgnn_conv import RelGNNConv
from rllm.nn.pre_encoder import HeteroTemporalEncoder


class RelGNN(torch.nn.Module):
    r"""The RelGNN model is a GNN framework specifically designed
    to leverage the unique structural characteristics of the graphs
    built from relational databases from paper
    `"RelGNN: Composite Message Passing for Relational Deep Learning"
    <https://arxiv.org/abs/2502.06784>`_ paper.

    Args:
        node_types (List[str]): The list of node types.
        atomic_routes_edge_types (List[Tuple[str, str, str]]): The list of edge types
            corresponding to atomic message passing routes.
        hidden_dims (int): The number of hidden dimensions.
        aggr (str): The aggregation method.
        num_layers (int): The number of layers.
        num_heads (int): The number of attention heads.
        simplified_MP (bool): Whether to use simplified message passing.
    """

    def __init__(
        self,
        node_types: List[str],
        atomic_routes_edge_types: List[Tuple],
        hidden_dim: int,
        aggr: str = "sum",
        num_layers: int = 2,
        num_heads: int = 1,
        simplified_MP=True,
    ):
        super().__init__()

        self.simplified_MP = simplified_MP
        self.aggr = aggr

        self.edge_type_mapping = {
            edge_type: "__".join(edge_type)
            for edge_type in atomic_routes_edge_types
        }

        convs = ModuleList()
        for _ in range(num_layers):
            conv_dict = ModuleDict()
            for edge_type in atomic_routes_edge_types:
                conv_dict[self.edge_type_mapping[edge_type]] = \
                    RelGNNConv(
                        attn_type=edge_type[0],
                        in_dim=hidden_dim,
                        out_dim=hidden_dim,
                        num_heads=num_heads,
                        aggr=aggr,
                        simplified_MP=simplified_MP,
                    )
            convs.append(conv_dict)
        self.convs = convs

        self.norms = ModuleList()
        for _ in range(num_layers):
            norm_dict = torch.nn.ModuleDict()
            for node_type in node_types:
                norm_dict[node_type] = torch.nn.LayerNorm(hidden_dim)
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
        for _, (conv_dict, norm_dict) in enumerate(zip(self.convs, self.norms)):
            x_dict =  self.heteroconv_forward(conv_dict, x_dict, edge_index_dict)
            x_dict = {key: norm_dict[key](x) for key, x in x_dict.items()}
            x_dict = {key: x.relu() for key, x in x_dict.items()}

        return x_dict

    def heteroconv_forward(
        self,
        conv_dict: ModuleDict,
        x_dict: Dict[str, Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], Tensor]
    ) -> Dict[str, Tensor]:

        out_dict: Dict[str, List[Tensor]] = defaultdict(list)

        # inner helper functions
        def update(out_dict, dst, out):
            if dst not in out_dict:
                out_dict[dst] = [out]
            else:
                out_dict[dst].append(out)

        def group(xs: List[Tensor], aggr: Optional[str]) -> Optional[Tensor]:
            if len(xs) == 0:
                return None
            elif aggr is None:
                return torch.stack(xs, dim=1)
            elif len(xs) == 1:
                return xs[0]
            elif aggr == "cat":
                return torch.cat(xs, dim=-1)
            else:
                out = torch.stack(xs, dim=0)
                out = getattr(torch, aggr)(out, dim=0)
                out = out[0] if isinstance(out, tuple) else out
                return out

        for edge_type_info, conv in conv_dict.items():
            edge_type_info = edge_type_info.split("__")
            attn_type = edge_type_info[0]

            if attn_type == 'dim-dim':
                src, rel, dst = edge_type_info[1:]
                x = (
                        x_dict.get(src, None),
                        x_dict.get(dst, None),
                    )
                edge_index = edge_index_dict[(src, rel, dst)]

                out = conv(x, edge_index)

                if self.simplified_MP and out is None:
                    continue

                update(out_dict, dst, out)

            elif attn_type == 'dim-fact-dim':
                edge_attn, edge_aggr = edge_type_info[1:4], edge_type_info[4:]
                edge_attn = tuple(edge_attn)
                edge_aggr = tuple(edge_aggr)
                src_attn, _, dst = edge_attn
                src_aggr = edge_aggr[0]
                x = (
                        x_dict[src_aggr],
                        x_dict[src_attn],
                        x_dict[dst],
                    )
                edge_index = (
                        edge_index_dict[edge_attn],
                        edge_index_dict[edge_aggr],
                    )
                out = conv(x, edge_index)

                if self.simplified_MP and out is None:
                    continue

                out_dst, out_src_attn = out
                update(out_dict, dst, out_dst)
                update(out_dict, src_attn, out_src_attn)

        for key, value in out_dict.items():
            out_dict[key] = group(value, self.aggr)

        if self.simplified_MP:
            for key, value in x_dict.items():
                if key not in out_dict:
                    out_dict[key] = value

        return out_dict


class RelGNNModel(torch.nn.Module):
    r"""The relational table learning model with RelGNN as the HGNN
    backbone from paper
    `"RelGNN: Composite Message Passing for Relational Deep Learning"
    <https://arxiv.org/abs/2502.06784>`_ paper.
    The replementation includes Table ResNet as TNN and
    RelGNN as the HGNN following
    the original paper with temporal encoding module.

    Args:
        data (HeteroGraphData): The heterogeneous graph data.
        col_stats_dict (Dict[str, Dict[ColType, List[Dict[StatType, Any]]]]):
            The column statistics dictionary for each table.
        atomic_routes_edge_types (List[Tuple[str, str, str]]): The list of edge types
            corresponding to atomic message passing routes.
        hidden_dim (int): The hidden dimension.
        out_dim (int): The output dimension.
        tnn_hidden_dim (int): The hidden dimension for TNN.
        tnn_num_layers (int): The number of layers for TNN.
        relgnn_aggr (str): The aggregation method for RelGNN.
        relgnn_num_layers (int): The number of layers for RelGNN.
        relgnn_num_heads (int): The number of attention heads for RelGNN.
        relgnn_simplified_MP (bool): Whether to use simplified message passing in RelGNN.
        use_temporal_encoder (bool): Whether to use temporal encoder.
    """

    def __init__(
        self,
        data: HeteroGraphData,
        col_stats_dict: Dict[str, Dict[ColType, List[Dict[StatType, Any]]]],
        atomic_routes_edge_types: List[Tuple[str, str, str]],
        hidden_dim: int,
        out_dim: int,
        # TNN args
        tnn_hidden_dim: int = 128,
        tnn_num_layers: int = 4,
        # HGNN args
        relgnn_aggr: str = "mean",
        relgnn_num_layers: int = 2,
        relgnn_num_heads: int = 1,
        relgnn_simplified_MP: bool = True,
        # Temporal Encoder args
        use_temporal_encoder: bool = True,
        # Output head args
        reg_task: bool = False,
    ):
        super().__init__()
        # validate input
        for node_type in data.node_types:
            assert node_type in col_stats_dict, \
                f"Node type {node_type} not found in col_stats_dict"

        # build modules
        self.TNN_DICT = ModuleDict(
            {
                node_type: TableResNet(
                    hidden_dim=tnn_hidden_dim,
                    out_dim=hidden_dim,
                    num_layers=tnn_num_layers,
                    metadata=col_stats_dict[node_type],
                ) for node_type in data.node_types
            }
        )

        self.use_temporal_encoder = use_temporal_encoder
        if use_temporal_encoder:
            self.TEMPORAL_ENCODER = HeteroTemporalEncoder(
                node_types=[
                    node_type
                    for node_type in data.node_types
                    if "time" in data[node_type]
                ],
                channels=hidden_dim,
            )

        self.RelGNN = RelGNN(
            node_types=data.node_types,
            atomic_routes_edge_types=atomic_routes_edge_types,
            hidden_dim=hidden_dim,
            aggr=relgnn_aggr,
            num_layers=relgnn_num_layers,
            num_heads=relgnn_num_heads,
            simplified_MP=relgnn_simplified_MP,
        )

        if reg_task:
            self.OUTPUT_HEAD = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, out_dim),
                torch.nn.GELU(),
                torch.nn.Linear(out_dim, out_dim),
            )
        else:
            self.OUTPUT_HEAD = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, out_dim),
                torch.nn.BatchNorm1d(out_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(out_dim, out_dim),
            )

        self.reset_parameters()

    def reset_parameters(self):
        for tnn in self.TNN_DICT.values():
            tnn.reset_parameters()
        if self.use_temporal_encoder:
            self.TEMPORAL_ENCODER.reset_parameters()
        self.RelGNN.reset_parameters()
        for layer in self.OUTPUT_HEAD:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(
        self,
        batch: HeteroGraphData,
        target_table: str,
    ) -> Dict[str, Tensor]:
        seed_time = batch[target_table].seed_time
        # 1. apply TNN to each node type (table)
        x_dict = {}
        for node_type, node_storage in batch.node_items():
            x_dict[node_type] = self.TNN_DICT[node_type](node_storage.table)

        # 2. (optional) apply TEMPORAL_ENCODER to each temporal node type
        if self.use_temporal_encoder:
            assert hasattr(batch, "time_dict")
            assert hasattr(batch, "batch_dict")
            rel_time_dict = self.TEMPORAL_ENCODER(
                seed_time, batch.time_dict, batch.batch_dict
            )
            for node_type, rel_time in rel_time_dict.items():
                x_dict[node_type] = x_dict[node_type] + rel_time

        # 3. apply RelGNN
        x_dict = self.RelGNN(
            x_dict,
            batch.edge_index_dict
        )

        # 4. apply OUTPUT_HEAD to target table
        return self.OUTPUT_HEAD(x_dict[target_table][: seed_time.size(0)])