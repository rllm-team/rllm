from typing import Any, Dict, List

import torch
from torch import Tensor
from torch.nn import ModuleDict

from rllm.types import ColType, StatType
from rllm.data import HeteroGraphData
from rllm.nn.models import TableResNet, HeteroSAGE
from rllm.nn.common import HeteroTemporalEncoder


class RDL(torch.nn.Module):
    r"""Relational Deep Learning (RDL) model from paper
    `"RelBench: A Benchmark for Deep Learning on Relational Databases"
    <https://arxiv.org/abs/2407.20060>`_ paper.
    The RDL model combines Table Neural Networks (TNNs)
    and Heterogeneous Graph Neural Networks (HGNNs)
    to effectively learn from multi-table relational data.
    We consistently use TableResNet as the TNN component
    and HeteroSAGE as the HGNN component following the
    original paper with temporal encoding module.

    Args:
        data (HeteroGraphData): The heterogeneous graph data.
        col_stats_dict (Dict[str, Dict[ColType, List[Dict[StatType, Any]]]]):
            The column statistics dictionary for each table.
        hidden_dim (int): The hidden dimension.
        out_dim (int): The output dimension.
        tnn_hidden_dim (int): The hidden dimension for TNN.
        tnn_num_layers (int): The number of layers for TNN.
        hgnn_aggr (str): The aggregation method for HGNN.
        hgnn_num_layers (int): The number of layers for HGNN.
        use_temporal_encoder (bool): Whether to use temporal encoder.
    """

    def __init__(
        self,
        data: HeteroGraphData,
        col_stats_dict: Dict[str, Dict[ColType, List[Dict[StatType, Any]]]],
        hidden_dim: int,
        out_dim: int,
        # TNN args
        tnn_hidden_dim: int = 128,
        tnn_num_layers: int = 4,
        # HGNN args
        hgnn_aggr: str = "mean",
        hgnn_num_layers: int = 2,
        # Temporal Encoder args
        use_temporal_encoder: bool = False,
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
        else:
            self.TEMPORAL_ENCODER = self.register_parameter(
                "TEMPORAL_ENCODER", None
            )

        self.HGNN = HeteroSAGE(
            node_types=data.node_types,
            edge_types=data.edge_types,
            hidden_dim=hidden_dim,
            aggr=hgnn_aggr,
            num_layers=hgnn_num_layers,
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
        self.HGNN.reset_parameters()
        for module in self.OUTPUT_HEAD.children():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()

    def forward(
        self,
        batch: HeteroGraphData,
        target_table: str,
    ) -> Tensor:
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

        # 3. apply HGNN
        x_dict = self.HGNN(
            x_dict,
            batch.edge_index_dict
        )

        # 4. apply OUTPUT_HEAD to target table
        return self.OUTPUT_HEAD(x_dict[target_table][: seed_time.size(0)])