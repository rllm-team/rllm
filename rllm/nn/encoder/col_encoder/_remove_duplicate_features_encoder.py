from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from torch import Tensor

from rllm.types import ColType, StatType
from ._col_encoder import ColEncoder


class RemoveDuplicateFeaturesEncoder(ColEncoder):
    r"""Bridge ColEncoder for duplicate-feature removal.

    Current behavior intentionally matches TabPFN's current implementation,
    which returns the input tensor directly.
    """

    supported_types = {ColType.NUMERICAL, ColType.CATEGORICAL, ColType.BINARY}

    def __init__(
        self,
        out_dim: Optional[int] = None,
        stats_list: Optional[List[Dict[StatType, Any]]] = None,
        post_module: Optional[torch.nn.Module] = None,
    ) -> None:
        super().__init__(
            out_dim=out_dim, stats_list=stats_list, post_module=post_module
        )

    def post_init(self) -> None:
        return None

    def reset_parameters(self) -> None:
        super().reset_parameters()

    def encode_forward(self, feat: Tensor) -> Tensor:
        out = feat
        if self.post_module is not None:
            out = self.post_module(out)
        return out
