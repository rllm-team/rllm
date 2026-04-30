from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from torch import Tensor

from rllm.types import ColType, StatType
from ._col_encoder import ColEncoder


class NanInfIndicatorEncoder(ColEncoder):
    r"""Encode NaN and signed infinity positions as TabPFN indicator values."""

    supported_types = {
        ColType.NUMERICAL,
        ColType.CATEGORICAL,
        ColType.BINARY,
    }

    def __init__(
        self,
        nan_indicator: float = -2.0,
        pos_inf_indicator: float = 2.0,
        neg_inf_indicator: float = 4.0,
        out_dim: Optional[int] = None,
        stats_list: Optional[List[Dict[StatType, Any]]] = None,
        post_module: Optional[torch.nn.Module] = None,
    ) -> None:
        super().__init__(
            out_dim=out_dim,
            stats_list=stats_list,
            post_module=post_module,
        )
        self.nan_indicator = float(nan_indicator)
        self.pos_inf_indicator = float(pos_inf_indicator)
        self.neg_inf_indicator = float(neg_inf_indicator)

    def post_init(self) -> None:
        return None

    def reset_parameters(self) -> None:
        super().reset_parameters()

    def encode_forward(self, feat: Tensor, **kwargs: object) -> Tensor:
        del kwargs
        out = (
            torch.isnan(feat) * self.nan_indicator
            + torch.logical_and(torch.isinf(feat), torch.sign(feat) == 1)
            * self.pos_inf_indicator
            + torch.logical_and(torch.isinf(feat), torch.sign(feat) == -1)
            * self.neg_inf_indicator
        ).to(feat.dtype)
        if self.post_module is not None:
            out = self.post_module(out)
        return out
