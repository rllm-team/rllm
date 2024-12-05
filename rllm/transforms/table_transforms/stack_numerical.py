from __future__ import annotations

import torch

from rllm.types import ColType, StatType
from rllm.data import TableData
from .base_transform import BaseTransform


class StackNumerical(BaseTransform):
    def __init__(
        self,
        out_dim: int,
    ) -> None:
        self.out_dim = out_dim

    def forward(
        self,
        data: TableData,
    ) -> TableData:
        if ColType.NUMERICAL in data.feat_dict.keys():
            metadata = data.metadata[ColType.NUMERICAL]
            self.mean = torch.tensor([stats[StatType.MEAN] for stats in metadata])
            self.std = torch.tensor([stats[StatType.STD] for stats in metadata]) + 1e-6

            feat = data.feat_dict[ColType.NUMERICAL]
            feat = (feat - self.mean) / self.std
            data.feat_dict[ColType.NUMERICAL] = feat.unsqueeze(2).repeat(
                1, 1, self.out_dim
            )
        return data
