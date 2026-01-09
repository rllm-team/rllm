from typing import Dict, List

import torch
from torch import Tensor
from torch.nn import ModuleDict

from ..pre_encoder.positional_encoder import PositionalEncoder


class HeteroTemporalEncoder(torch.nn.Module):
    r"""HeteroTemporalEncoder for RDL model.

    Args:
        node_types (List[str]): The list of node types.
        channels (int): The number of channels.
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
        out_dict: Dict[str, Tensor] = {}

        for node_type, time in time_dict.items():
            rel_time = seed_time[batch_dict[node_type]] - time
            rel_time = rel_time / (60 * 60 * 24)  # Convert seconds to days.

            x = self.encoder_dict[node_type](rel_time)
            x = self.lin_dict[node_type](x)
            out_dict[node_type] = x

        return out_dict