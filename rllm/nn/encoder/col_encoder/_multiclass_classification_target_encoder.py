from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from torch import Tensor

from rllm.types import ColType, StatType
from ._col_encoder import ColEncoder


class MulticlassClassificationTargetEncoder(ColEncoder):
    r"""ColEncoder counterpart of TabPFN multiclass target flattening step."""

    supported_types = {ColType.CATEGORICAL, ColType.BINARY}

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

    @staticmethod
    def _flatten_column(col: Tensor) -> Tensor:
        valid = col[~torch.isnan(col)]
        if valid.numel() == 0:
            return col
        unique_vals = torch.unique(valid)
        mapped = (col.unsqueeze(-1) > unique_vals).sum(dim=-1)
        mapped = mapped.to(col.dtype)
        mapped[torch.isnan(col)] = torch.nan
        return mapped

    def encode_forward(self, feat: Tensor) -> Tensor:
        if feat.ndim == 3 and feat.shape[-1] == 1:
            x = feat.squeeze(-1)
            squeeze_back = True
        elif feat.ndim == 2:
            x = feat
            squeeze_back = False
        else:
            raise ValueError(
                f"Expected feat shape [B,C] or [B,C,1], got {tuple(feat.shape)}"
            )

        out = x.clone()
        for col_idx in range(x.shape[1]):
            out[:, col_idx] = self._flatten_column(x[:, col_idx])

        if squeeze_back:
            out = out.unsqueeze(-1)

        if self.post_module is not None:
            out = self.post_module(out)
        return out
