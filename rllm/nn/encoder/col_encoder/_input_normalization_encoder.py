from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from torch import Tensor

from rllm.types import ColType, StatType
from rllm.nn.models.tabpfn_v2.encoders import normalize_data, remove_outliers
from ._col_encoder import ColEncoder


class InputNormalizationEncoder(ColEncoder):
    r"""Normalize numerical columns with per-column metadata statistics."""

    supported_types = {ColType.NUMERICAL}

    def __init__(
        self,
        out_dim: Optional[int] = None,
        stats_list: Optional[List[Dict[StatType, Any]]] = None,
        post_module: Optional[torch.nn.Module] = None,
        clip_range: tuple[float, float] = (-100.0, 100.0),
        normalize_x: bool = True,
        remove_outliers: bool = False,
    ) -> None:
        super().__init__(
            out_dim=out_dim, stats_list=stats_list, post_module=post_module
        )
        self.clip_range = clip_range
        self.normalize_x = normalize_x
        self.remove_outliers = remove_outliers

    def post_init(self) -> None:
        means = []
        stds = []
        mins = []
        maxs = []
        for stats in self.stats_list:
            means.append(float(stats.get(StatType.MEAN, 0.0)))
            stds.append(float(stats.get(StatType.STD, 1.0)))
            mins.append(float(stats.get(StatType.MIN, -float("inf"))))
            maxs.append(float(stats.get(StatType.MAX, float("inf"))))
        self.register_buffer("mean", torch.tensor(means))
        self.register_buffer("std", torch.tensor(stds) + 1e-20)
        self.register_buffer("min_val", torch.tensor(mins))
        self.register_buffer("max_val", torch.tensor(maxs))

    def reset_parameters(self) -> None:
        super().reset_parameters()

    def transform_tabpfn(
        self,
        feat: Tensor,
        *,
        single_eval_pos: int,
        normalize_on_train_only: bool,
    ) -> Tensor:
        normalize_position = single_eval_pos if normalize_on_train_only else -1
        x = feat
        if self.remove_outliers:
            x, _ = remove_outliers(
                x,
                normalize_positions=normalize_position,
            )
        if self.normalize_x:
            x = normalize_data(
                x,
                normalize_positions=normalize_position,
            )
        if self.post_module is not None:
            x = self.post_module(x)
        return x

    def encode_forward(self, feat: Tensor) -> Tensor:
        if feat.ndim == 2:
            x = feat
            if self.remove_outliers:
                x = torch.maximum(x, self.min_val.to(x.device))
                x = torch.minimum(x, self.max_val.to(x.device))
            if self.normalize_x:
                x = (x - self.mean.to(x.device)) / self.std.to(x.device)
            x = torch.clamp(x, min=self.clip_range[0], max=self.clip_range[1])
            out = x
        elif feat.ndim == 3:
            x = feat
            min_val = self.min_val.to(x.device).unsqueeze(-1)
            max_val = self.max_val.to(x.device).unsqueeze(-1)
            mean = self.mean.to(x.device).unsqueeze(-1)
            std = self.std.to(x.device).unsqueeze(-1)
            if self.remove_outliers:
                x = torch.maximum(x, min_val)
                x = torch.minimum(x, max_val)
            if self.normalize_x:
                x = (x - mean) / std
            x = torch.clamp(x, min=self.clip_range[0], max=self.clip_range[1])
            out = x
        else:
            raise ValueError(
                f"Expected feat to be 2D or 3D, but got shape {tuple(feat.shape)}."
            )

        if self.post_module is not None:
            out = self.post_module(out)

        return out
