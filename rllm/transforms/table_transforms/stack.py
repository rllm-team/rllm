from __future__ import annotations
from typing import Any, Dict, List

from rllm.types import ColType
from rllm.data import TableData
from .base_transform import BaseTransform


class StackNumerical(BaseTransform):
    def __init__(
        self,
        out_dim: int,
    ) -> None:
        self.out_dim = out_dim

    def forward(self, data: TableData) -> TableData:
        assert ColType.NUMERICAL in data.feat_dict.keys()
        feat = data.feat_dict[ColType.NUMERICAL]
        data.feat_dict[ColType.NUMERICAL] = feat.unsqueeze(2).repeat(1, 1, self.out_dim)
        return data
