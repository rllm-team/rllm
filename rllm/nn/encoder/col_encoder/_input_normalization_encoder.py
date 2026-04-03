from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, overload

import numpy as np
import torch
from torch import Tensor

from rllm.types import ColType, StatType
from ._col_encoder import ColEncoder


def torch_nansum(
    x: torch.Tensor,
    axis: int | tuple[int, ...] | None = None,
    keepdim: bool = False,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    nan_mask = torch.isnan(x)
    masked_input = torch.where(
        nan_mask,
        torch.tensor(0.0, device=x.device, dtype=x.dtype),
        x,
    )
    return torch.sum(masked_input, axis=axis, keepdim=keepdim, dtype=dtype)


@overload
def torch_nanmean(
    x: torch.Tensor,
    axis: int = 0,
    *,
    return_nanshare: Literal[False] = False,
    include_inf: bool = False,
) -> torch.Tensor: ...


@overload
def torch_nanmean(
    x: torch.Tensor,
    axis: int = 0,
    *,
    return_nanshare: Literal[True],
    include_inf: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]: ...


def torch_nanmean(
    x: torch.Tensor,
    axis: int = 0,
    *,
    return_nanshare: bool = False,
    include_inf: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    nan_mask = torch.isnan(x)
    if include_inf:
        nan_mask = torch.logical_or(nan_mask, torch.isinf(x))

    num = torch.where(nan_mask, torch.full_like(x, 0), torch.full_like(x, 1)).sum(  # type: ignore
        axis=axis,
    )
    value = torch.where(nan_mask, torch.full_like(x, 0), x).sum(axis=axis)  # type: ignore
    if return_nanshare:
        return value / num, 1.0 - (num / x.shape[axis])
    return value / num.clip(min=1.0)


def torch_nanstd(x: torch.Tensor, axis: int = 0) -> torch.Tensor:
    num = torch.where(torch.isnan(x), torch.full_like(x, 0), torch.full_like(x, 1)).sum(  # type: ignore
        axis=axis,
    )
    value = torch.where(torch.isnan(x), torch.full_like(x, 0), x).sum(axis=axis)  # type: ignore
    mean = value / num.clip(min=1.0)
    mean_broadcast = torch.repeat_interleave(
        mean.unsqueeze(axis),
        x.shape[axis],
        dim=axis,
    )
    var = torch_nansum(torch.square(mean_broadcast - x), axis=axis) / (num - 1).clip(
        min=1.0
    )
    return torch.sqrt(var)


@overload
def normalize_data(
    data: torch.Tensor,
    *,
    normalize_positions: int = -1,
    return_scaling: Literal[False] = False,
    clip: bool = True,
    std_only: bool = False,
    mean: torch.Tensor | None = None,
    std: torch.Tensor | None = None,
) -> torch.Tensor: ...


@overload
def normalize_data(
    data: torch.Tensor,
    *,
    normalize_positions: int = -1,
    return_scaling: Literal[True],
    clip: bool = True,
    std_only: bool = False,
    mean: torch.Tensor | None = None,
    std: torch.Tensor | None = None,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]: ...


def normalize_data(
    data: torch.Tensor,
    *,
    normalize_positions: int = -1,
    return_scaling: bool = False,
    clip: bool = True,
    std_only: bool = False,
    mean: torch.Tensor | None = None,
    std: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    assert (mean is None) == (
        std is None
    ), "Either both or none of mean and std must be given"
    if mean is None:
        if normalize_positions is not None and normalize_positions > 0:
            mean = torch_nanmean(data[:normalize_positions], axis=0)  # type: ignore
            std = torch_nanstd(data[:normalize_positions], axis=0)
        else:
            mean = torch_nanmean(data, axis=0)  # type: ignore
            std = torch_nanstd(data, axis=0)

        std = torch.where(std == 0, torch.ones_like(std), std)

        if len(data) == 1 or normalize_positions == 1:
            std = torch.ones_like(std)

        if std_only:
            mean = torch.zeros_like(mean)

    data = (data - mean) / (std + 1e-16)

    if clip:
        data = torch.clip(data, min=-100, max=100)

    if return_scaling:
        return data, (mean, std)
    return data


def remove_outliers(
    X: torch.Tensor,
    n_sigma: float = 4,
    normalize_positions: int = -1,
    lower: None | torch.Tensor = None,
    upper: None | torch.Tensor = None,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    assert (lower is None) == (upper is None), "Either both or none of lower and upper"
    assert len(X.shape) == 3, "X must be T,B,H"
    if lower is None:
        data = X if normalize_positions == -1 else X[:normalize_positions]
        data_clean = data[:].clone()
        data_mean, data_std = torch_nanmean(data, axis=0), torch_nanstd(data, axis=0)
        cut_off = data_std * n_sigma
        lower, upper = data_mean - cut_off, data_mean + cut_off

        data_clean[torch.logical_or(data_clean > upper, data_clean < lower)] = np.nan
        data_mean, data_std = (
            torch_nanmean(data_clean, axis=0),
            torch_nanstd(data_clean, axis=0),
        )
        cut_off = data_std * n_sigma
        lower, upper = data_mean - cut_off, data_mean + cut_off

    X = torch.maximum(-torch.log(1 + torch.abs(X)) + lower, X)
    X = torch.minimum(torch.log(1 + torch.abs(X)) + upper, X)
    return X, (lower, upper)


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
        # This encoder is intended to be statistics-driven (non-learnable).
        # If a post module is provided, freeze it to avoid introducing trainable
        # parameters through normalization.
        if self.post_module is not None:
            for p in self.post_module.parameters():
                p.requires_grad_(False)

    def post_init(self) -> None:
        # Keep API compatibility with ColEncoder lifecycle, but do not
        # materialize per-column stats as buffers to avoid state_dict coupling.
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
        normalize_position = (
            single_eval_pos
            if (single_eval_pos is not None and normalize_on_train_only)
            else -1
        )
        if feat.ndim == 2:
            x = feat
            if self.remove_outliers:
                data_for_stats = (
                    x[:normalize_position]
                    if normalize_position is not None and normalize_position > 0
                    else x
                )
                data_mean = torch_nanmean(data_for_stats, axis=0)  # type: ignore
                data_std = torch_nanstd(data_for_stats, axis=0)
                cut_off = data_std * 4.0
                lower, upper = data_mean - cut_off, data_mean + cut_off
                x = torch.maximum(-torch.log(1 + torch.abs(x)) + lower, x)
                x = torch.minimum(torch.log(1 + torch.abs(x)) + upper, x)
            if self.normalize_x:
                x = normalize_data(
                    x,
                    normalize_positions=normalize_position,
                )
            x = torch.clamp(x, min=self.clip_range[0], max=self.clip_range[1])
            out = x
        elif feat.ndim == 3:
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
            x = torch.clamp(x, min=self.clip_range[0], max=self.clip_range[1])
            out = x
        else:
            raise ValueError(
                f"Expected feat to be 2D or 3D, but got shape {tuple(feat.shape)}."
            )

        if self.post_module is not None:
            out = self.post_module(out)

        return out
