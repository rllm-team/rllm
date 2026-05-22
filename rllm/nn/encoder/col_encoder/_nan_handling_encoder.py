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
        # Keep lifecycle compatibility, but avoid stateful buffers so this
        # encoder contributes no normalization metadata keys to state_dict.
        return None

    def reset_parameters(self) -> None:
        super().reset_parameters()

    def encode_forward(
        self,
        feat: Tensor,
        *,
        single_eval_pos: Optional[int] = None,
        normalize_on_train_only: bool = True,
    ) -> Tensor:
        def _compute_fill_values(x: Tensor) -> Tensor:
            invalid = torch.isnan(x) | torch.isinf(x)
            valid = (~invalid).to(x.dtype).sum(dim=0).clamp(min=1.0)
            value = torch.where(invalid, torch.zeros_like(x), x).sum(dim=0)
            return value / valid

        if feat.ndim == 2:
            fill = _compute_fill_values(feat)
            mask = torch.isnan(feat) | torch.isinf(feat)
            out = feat.clone()
            out[mask] = fill.unsqueeze(0).expand_as(out)[mask]
        elif feat.ndim == 3:
            stats_source = (
                feat[:single_eval_pos]
                if (single_eval_pos is not None and normalize_on_train_only)
                else feat
            )
            fill = _compute_fill_values(stats_source)
            mask = torch.isnan(feat) | torch.isinf(feat)
            out = feat.clone()
            out[mask] = fill.unsqueeze(0).expand_as(out)[mask]
        else:
            raise ValueError(
                f"Expected feat to be 2D or 3D, but got shape {tuple(feat.shape)}."
            )

        return out
