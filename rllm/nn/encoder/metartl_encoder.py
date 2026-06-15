from typing import Any, Dict, List

import torch
from torch import Tensor
from torch.nn import ModuleDict

from rllm.types import ColType, StatType
from rllm.data import HeteroGraphData
from rllm.nn.models.resnet import TableResNet
from rllm.nn.models.heterosage import HeteroSAGE
from rllm.nn.models.relgnn import RelGNN
from rllm.nn.encoder.heterotemporal_encoder import HeteroTemporalEncoder
from rllm.utils import get_atomic_routes


class MetaRTLEncoder(torch.nn.Module):
    r"""Stage-one relational table encoder of MetaRTL
    `"MetaRTL: Meta-path Attention Enhanced Relational Table Learning"`.

    It encodes each table with a Table Neural Network (TableResNet), optionally
    adds relative temporal embeddings, and propagates information with a
    heterogeneous GNN backbone. The model can be used directly for prediction
    (:meth:`forward`) during pretraining, and its intermediate node embeddings
    (:meth:`encode`) feed the meta-path propagation of stage two.

    The GNN backbone is swappable via :obj:`gnn`: it supports the GraphSAGE
    backbone of :class:`~rllm.nn.models.RDL` and the
    :class:`~rllm.nn.models.RelGNN` backbone.

    Args:
        data (HeteroGraphData): The heterogeneous graph data.
        col_stats_dict (Dict[str, Dict[ColType, List[Dict[StatType, Any]]]]):
            The column statistics dictionary for each table.
        hidden_dim (int): The hidden dimension.
        out_dim (int): The output dimension.
        gnn (str): The HGNN backbone, one of :obj:`"SAGE"` or :obj:`"RelGNN"`.
            (default: :obj:`"SAGE"`)
        tnn_hidden_dim (int): The hidden dimension for TNN. (default: :obj:`128`)
        tnn_num_layers (int): The number of layers for TNN. (default: :obj:`4`)
        gnn_aggr (str): The aggregation method for the HGNN.
            (default: :obj:`"sum"`)
        gnn_num_layers (int): The number of HGNN layers. (default: :obj:`2`)
        gnn_num_heads (int): The number of attention heads (RelGNN only).
            (default: :obj:`1`)
        use_temporal_encoder (bool): Whether to use the temporal encoder.
            (default: :obj:`True`)
        reg_task (bool): If :obj:`True`, uses a regression output head with
            :class:`~torch.nn.GELU` activation. (default: :obj:`False`)

    Example:
        >>> from rllm.nn.encoder import MetaRTLEncoder
        >>> model = MetaRTLEncoder(
        ...     data=hdata,
        ...     col_stats_dict=col_stats_dict,
        ...     hidden_dim=128,
        ...     out_dim=1,
        ...     gnn="SAGE",
        ... )
    """

    def __init__(
        self,
        data: HeteroGraphData,
        col_stats_dict: Dict[str, Dict[ColType, List[Dict[StatType, Any]]]],
        hidden_dim: int,
        out_dim: int,
        gnn: str = "SAGE",
        # TNN args
        tnn_hidden_dim: int = 128,
        tnn_num_layers: int = 4,
        # HGNN args
        gnn_aggr: str = "sum",
        gnn_num_layers: int = 2,
        gnn_num_heads: int = 1,
        # Temporal Encoder args
        use_temporal_encoder: bool = True,
        # Output head args
        reg_task: bool = False,
    ):
        super().__init__()
        # validate input
        for node_type in data.node_types:
            assert (
                node_type in col_stats_dict
            ), f"Node type {node_type} not found in col_stats_dict"

        self.gnn_name = gnn

        # table encoders (the embedding layer)
        self.TNN_DICT = ModuleDict(
            {
                node_type: TableResNet(
                    hidden_dim=tnn_hidden_dim,
                    out_dim=hidden_dim,
                    num_layers=tnn_num_layers,
                    metadata=col_stats_dict[node_type],
                )
                for node_type in data.node_types
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
            self.TEMPORAL_ENCODER = self.register_parameter("TEMPORAL_ENCODER", None)

        # swappable HGNN backbone
        if gnn == "SAGE":
            self.HGNN = HeteroSAGE(
                node_types=data.node_types,
                edge_types=data.edge_types,
                hidden_dim=hidden_dim,
                aggr=gnn_aggr,
                num_layers=gnn_num_layers,
            )
        elif gnn == "RelGNN":
            self.HGNN = RelGNN(
                node_types=data.node_types,
                atomic_routes_edge_types=get_atomic_routes(data.edge_types),
                hidden_dim=hidden_dim,
                aggr=gnn_aggr,
                num_layers=gnn_num_layers,
                num_heads=gnn_num_heads,
            )
        else:
            raise ValueError(f"Unrecognized gnn backbone {gnn} for MetaRTLEncoder")

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
        r"""Resets all learnable parameters of the module."""
        for tnn in self.TNN_DICT.values():
            tnn.reset_parameters()
        if self.use_temporal_encoder:
            self.TEMPORAL_ENCODER.reset_parameters()
        self.HGNN.reset_parameters()
        for layer in self.OUTPUT_HEAD:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

    def encode(self, batch: HeteroGraphData) -> Dict[str, Tensor]:
        r"""Encode a mini-batch into per-node-type embeddings.

        Runs the table encoders, optional temporal encoder, and the HGNN
        backbone, returning the node embedding dict used for meta-path
        propagation in stage two.

        Args:
            batch (HeteroGraphData): Batched heterogeneous relational graph.

        Returns:
            Dict[str, Tensor]: Node embeddings by node type.
        """
        x_dict = {}
        for node_type, node_storage in batch.node_items():
            x_dict[node_type] = self.TNN_DICT[node_type](node_storage.table)

        if self.use_temporal_encoder:
            seed_time = batch[self._target_table(batch)].seed_time
            rel_time_dict = self.TEMPORAL_ENCODER(
                seed_time, batch.time_dict, batch.batch_dict
            )
            for node_type, rel_time in rel_time_dict.items():
                x_dict[node_type] = x_dict[node_type] + rel_time

        x_dict = self.HGNN(x_dict, batch.edge_index_dict)
        return x_dict

    @staticmethod
    def _target_table(batch: HeteroGraphData) -> str:
        # The entity table is the only node type carrying ``seed_time``.
        for node_type, node_storage in batch.node_items():
            if "seed_time" in node_storage:
                return node_type
        raise ValueError("No target table with seed_time found in batch.")

    def forward(
        self,
        batch: HeteroGraphData,
        target_table: str,
    ) -> Tensor:
        r"""Run table encoding, optional temporal encoding, HGNN propagation,
        and the output head.

        Args:
            batch (HeteroGraphData): Batched heterogeneous relational graph.
            target_table (str): The node type to predict.

        Returns:
            Tensor: Output predictions for the target table. For temporal
            entity tasks, only the seed nodes are returned; for transductive
            full-graph settings (no :obj:`seed_time`), predictions for all
            target nodes are returned.
        """
        x_dict = self.encode(batch)
        x = x_dict[target_table]
        if "seed_time" in batch[target_table]:
            x = x[: batch[target_table].seed_time.size(0)]
        return self.OUTPUT_HEAD(x)
