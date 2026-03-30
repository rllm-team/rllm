from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from torch import Tensor

from rllm.types import ColType, StatType
from ._col_encoder import ColEncoder


class NanHandlingEncoder(ColEncoder):
    r"""Replace NaN/Inf values using per-column metadata statistics."""

    supported_types = {ColType.NUMERICAL}

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
        means = []
        for stats in self.stats_list:
            mean = stats.get(StatType.MEAN, 0.0)
            means.append(float(mean))
        self.register_buffer("fill_values", torch.tensor(means))

    def reset_parameters(self) -> None:
        super().reset_parameters()

    def encode_forward(
        self,
        feat: Tensor,
        *,
        single_eval_pos: Optional[int] = None,
        normalize_on_train_only: bool = True,
    ) -> Tensor:
        if feat.ndim == 2:
            fill = self.fill_values.to(feat.device)
            mask = torch.isnan(feat) | torch.isinf(feat)
            out = feat.clone()
            out[mask] = fill.unsqueeze(0).expand_as(out)[mask]
        elif feat.ndim == 3:
            if single_eval_pos is not None:
                train_x = feat[:single_eval_pos]
                fill = torch.nanmean(train_x, dim=0)
                mask = torch.isnan(feat) | torch.isinf(feat)
                out = feat.clone()
                out[mask] = fill.unsqueeze(0).expand_as(out)[mask]
            else:
                fill = self.fill_values.to(feat.device).unsqueeze(-1)
                mask = torch.isnan(feat) | torch.isinf(feat)
                out = feat.clone()
                out[mask] = fill.unsqueeze(0).expand_as(out)[mask]
        else:
            raise ValueError(
                f"Expected feat to be 2D or 3D, but got shape {tuple(feat.shape)}."
            )

        return out
