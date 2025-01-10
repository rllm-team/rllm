from __future__ import annotations

import torch

from rllm.types import ColType, StatType
from rllm.data import TableData
from .col_transform import ColTransform


class ColNormalize(ColTransform):
    r"""The ColNormalize class is designed to normalize numerical features
    in tabular data. This transformation standardizes the numerical features by
    subtracting the mean and dividing by the standard deviation.
    """

    def __init__(
        self,
    ) -> None:
        super().__init__()

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

            data.feat_dict[ColType.NUMERICAL] = feat

        return data
