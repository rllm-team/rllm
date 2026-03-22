from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from torch import Tensor

from rllm.types import ColType, StatType
from ._col_encoder import ColEncoder


class VariableNumFeaturesEncoder(ColEncoder):
    r"""Map variable-width feature blocks to a fixed number of columns."""

    supported_types = {ColType.NUMERICAL, ColType.CATEGORICAL, ColType.BINARY}

    def __init__(
        self,
        num_features: int,
        normalize_by_used_features: bool = True,
        normalize_by_sqrt: bool = True,
        out_dim: Optional[int] = None,
        stats_list: Optional[List[Dict[StatType, Any]]] = None,
        post_module: Optional[torch.nn.Module] = None,
    ) -> None:
        super().__init__(
            out_dim=out_dim, stats_list=stats_list, post_module=post_module
        )
        self.num_features = num_features
        self.normalize_by_used_features = normalize_by_used_features
        self.normalize_by_sqrt = normalize_by_sqrt

    def post_init(self) -> None:
        return None

    def reset_parameters(self) -> None:
        super().reset_parameters()

    def transform_tabpfn(
        self,
        feat: Tensor,
        *,
        single_eval_pos: int,
    ) -> Tensor:
        x = feat
        if x.shape[2] == 0:
            out = torch.zeros(
                x.shape[0],
                x.shape[1],
                self.num_features,
                device=x.device,
                dtype=x.dtype,
            )
            if self.post_module is not None:
                out = self.post_module(out)
            return out

        train_x = x[:single_eval_pos]
        sel = (train_x[1:] == train_x[0]).sum(0) != (train_x.shape[0] - 1)
        used_feature_count = torch.clip(
            sel.sum(-1).unsqueeze(-1),
            min=1,
        ).to(x.device)

        if self.normalize_by_used_features:
            scale = self.num_features / used_feature_count
            if self.normalize_by_sqrt:
                scale = torch.sqrt(scale)
            x = x * scale

        zeros = torch.zeros(
            *x.shape[:-1],
            self.num_features - x.shape[-1],
            device=x.device,
            dtype=x.dtype,
        )
        out = torch.cat([x, zeros], dim=-1)
        if self.post_module is not None:
            out = self.post_module(out)
        return out

    def _estimate_used_feature_count(self, feat: Tensor) -> int:
        if feat.ndim == 2:
            sel = (feat[1:] == feat[0]).sum(0) != (feat.shape[0] - 1)
        elif feat.ndim == 3:
            sel = (feat[1:] == feat[0]).all(-1).sum(0) != (feat.shape[0] - 1)
        else:
            raise ValueError(
                f"Expected feat to be 2D or 3D, but got shape {tuple(feat.shape)}."
            )
        return max(int(sel.sum().item()), 1)

    def encode_forward(self, feat: Tensor) -> Tensor:
        used_feature_count = self._estimate_used_feature_count(feat)

        x = feat
        in_features = x.shape[1]

        if self.normalize_by_used_features:
            scale = self.num_features / used_feature_count
            if self.normalize_by_sqrt:
                scale = scale**0.5
            x = x * scale

        if in_features > self.num_features:
            if x.ndim == 2:
                x = x[:, : self.num_features]
            else:
                x = x[:, : self.num_features, :]
        elif in_features < self.num_features:
            pad_cols = self.num_features - in_features
            if x.ndim == 2:
                zeros = torch.zeros(
                    x.shape[0], pad_cols, device=x.device, dtype=x.dtype
                )
            else:
                zeros = torch.zeros(
                    x.shape[0], pad_cols, x.shape[2], device=x.device, dtype=x.dtype
                )
            x = torch.cat([x, zeros], dim=1)

        if self.post_module is not None:
            x = self.post_module(x)
        return x
