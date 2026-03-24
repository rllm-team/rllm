from typing import Dict, List

import torch
from torch import Tensor
from torch.nn import ModuleDict

from .col_encoder._positional_encoder import PositionalEncoder


class HeteroTemporalEncoder(torch.nn.Module):
    r"""HeteroTemporalEncoder for RDL model from paper
    `"RelBench: A Benchmark for Deep Learning on Relational Databases"
    <https://arxiv.org/abs/2407.20060>`_.

    Args:
        node_types (List[str]): The list of node types.
        channels (int): The number of channels.

    Example:
        >>> import torch
        >>> enc = HeteroTemporalEncoder(node_types=["user", "item"], channels=16)
        >>> seed_time = torch.tensor([1000.0, 1100.0])
        >>> time_dict = {"user": torch.tensor([900.0]), "item": torch.tensor([950.0])}
        >>> batch_dict = {"user": torch.tensor([0]), "item": torch.tensor([1])}
        >>> out = enc(seed_time, time_dict, batch_dict)
        >>> out["user"].shape
        torch.Size([1, 16])
    """

    def __init__(self, node_types: List[str], channels: int):
        super().__init__()

        self.encoder_dict = ModuleDict(
            {node_type: PositionalEncoder(channels) for node_type in node_types}
        )
        self.lin_dict = ModuleDict(
            {node_type: torch.nn.Linear(channels, channels) for node_type in node_types}
        )

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        for encoder in self.encoder_dict.values():
            encoder.reset_parameters()
        for lin in self.lin_dict.values():
            lin.reset_parameters()

    def forward(
        self,
        seed_time: Tensor,
        time_dict: Dict[str, Tensor],
        batch_dict: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        r"""Compute relative temporal embeddings for each node type.

        Args:
            seed_time (Tensor): The reference timestamps for seed nodes of
                shape :obj:`[num_seeds]`.
            time_dict (Dict[str, Tensor]): Timestamps per node type.
            batch_dict (Dict[str, Tensor]): Batch assignment indices per
                node type, mapping each node to a seed node.

        Returns:
            Dict[str, Tensor]: Temporal embeddings per node type of shape
            :obj:`[num_nodes, channels]`.
        """
        out_dict: Dict[str, Tensor] = {}

        for node_type, time in time_dict.items():
            rel_time = seed_time[batch_dict[node_type]] - time
            rel_time = rel_time / (60 * 60 * 24)  # Convert seconds to days.

            x = self.encoder_dict[node_type](rel_time)
            x = self.lin_dict[node_type](x)
            out_dict[node_type] = x

        return out_dict
