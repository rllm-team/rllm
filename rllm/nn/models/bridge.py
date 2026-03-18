from typing import List, Union

import torch
from torch import Tensor

from rllm.data import TableData
from rllm.nn.encoder import TableEncoder, GraphEncoder


class BRIDGE(torch.nn.Module):
    r"""The BRIDGE model introduced in the `"rLLM: Relational Table Learning
    with LLMs" <https://arxiv.org/abs/2407.20157>`__ paper.
    BRIDGE is a simple RTL method based on rLLM framework, which
    combines table neural networks (TNNs) and graph neural networks (GNNs) to
    deal with multi-table data and their interrelationships, and uses "foreign
    keys" to build relationships and analyze them to improve the performance of
    multi-table joint learning tasks.

    Args:
        table_backbone (TableBackbone): Backbones for tabular data.
        graph_backbone (GraphBackbone): Backbones for graph data.

    Returns:
        This class does not return tensors in ``__init__``.
        The ``forward`` method returns node representations for table rows.

    Example:
        >>> from rllm.nn.models.bridge import BRIDGE, TableBackbone, GraphBackbone
        >>> model = BRIDGE(TableBackbone(16, 32, metadata={}), GraphBackbone(32, 8))
    """

    def __init__(
        self,
        table_backbone: TableEncoder,
        graph_backbone: GraphEncoder,
    ) -> None:
        super().__init__()
        self.table_backbone = table_backbone
        self.graph_backbone = graph_backbone

    def forward(
        self,
        table: TableData,
        non_table: Tensor,
        adj: Union[Tensor, List[Tensor]],
    ) -> Tensor:
        """
        First, the Table Neural Network (TNN) learns the tabular data.
        Second, the learned representations are concatenated with the non-tabular data.
        Third, the Graph Neural Network (GNN) processes the combined data.
        along with the adjacency matrix to learn the overall representation.

        Args:
            table (Tensor): Input tabular data.
            non_table (Tensor): Input non-tabular data.
            adj (Tensor): Adjacency matrix.

        Returns:
            Tensor: Output table embedding features.
        """
        t_embedds = self.table_backbone(table)
        if isinstance(t_embedds, dict):
            t_embedds = torch.cat(list(t_embedds.values()), dim=0)
            t_embedds = t_embedds.mean(dim=1)
        if non_table is not None:
            node_feats = torch.cat([t_embedds, non_table], dim=0)
        else:
            node_feats = t_embedds
        node_feats = self.graph_backbone(node_feats, adj)
        return node_feats[: len(table), :]
