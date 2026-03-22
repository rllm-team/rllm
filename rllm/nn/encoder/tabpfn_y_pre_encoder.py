from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class TabPFNYPreEncoder(nn.Module):
    r"""Pre-encoder for TabPFN target tensor ``y``.

    This module mirrors the old ``get_y_encoder`` step-chain behavior while
    exposing a single ``nn.Module`` interface.
    """

    nan_indicator = -2.0
    inf_indicator = 2.0
    neg_inf_indicator = 4.0

    def __init__(
        self,
        *,
        num_inputs: int,
        embedding_size: int,
        nan_handling_y_encoder: bool,
        max_num_classes: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.num_inputs = num_inputs
        self.embedding_size = embedding_size
        self.nan_handling_y_encoder = nan_handling_y_encoder
        self.max_num_classes = max_num_classes

        in_features = num_inputs + (num_inputs if nan_handling_y_encoder else 0)
        self.proj = nn.Linear(in_features, embedding_size, bias=bias)

    @staticmethod
    def _train_prefix(y: torch.Tensor, single_eval_pos: Optional[int]) -> torch.Tensor:
        if single_eval_pos is None:
            return y
        return y[:single_eval_pos]

    def _compute_nan_indicators(self, y: torch.Tensor) -> torch.Tensor:
        return (
            torch.isnan(y) * self.nan_indicator
            + torch.logical_and(torch.isinf(y), torch.sign(y) == 1) * self.inf_indicator
            + torch.logical_and(torch.isinf(y), torch.sign(y) == -1)
            * self.neg_inf_indicator
        ).to(y.dtype)

    def _nan_handle(
        self, y: torch.Tensor, single_eval_pos: Optional[int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        y_prefix = self._train_prefix(y, single_eval_pos)
        feature_means = torch.nanmean(y_prefix, dim=0)

        indicators = self._compute_nan_indicators(y)
        mask = torch.isnan(y) | torch.isinf(y)
        y_out = y.clone()
        y_out[mask] = feature_means.unsqueeze(0).expand_as(y_out)[mask]
        return y_out, indicators

    @staticmethod
    def _flatten_targets(
        col: torch.Tensor, unique_vals: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if unique_vals is None:
            unique_vals = torch.unique(col)
        return (col.unsqueeze(-1) > unique_vals).sum(dim=-1)

    def _multiclass_encode(
        self, y: torch.Tensor, single_eval_pos: Optional[int]
    ) -> torch.Tensor:
        if y.ndim != 3 or y.shape[-1] != 1:
            raise ValueError(f"Expected y shape [S,B,1], got {tuple(y.shape)}")

        y_out = y.clone()
        y_prefix = self._train_prefix(y_out, single_eval_pos)
        for b in range(y_out.shape[1]):
            unique_vals = torch.unique(y_prefix[:, b])
            y_out[:, b, :] = self._flatten_targets(y_out[:, b, :], unique_vals)
        return y_out

    def forward(
        self,
        y_state: dict,
        single_eval_pos: Optional[int] = None,
        cache_trainset_representation: bool = False,
        **_: object,
    ) -> torch.Tensor:
        del cache_trainset_representation
        y = y_state["main"]

        inputs = []

        if self.nan_handling_y_encoder:
            y, indicators = self._nan_handle(y, single_eval_pos)
            inputs.append(y)
            inputs.append(indicators)
        else:
            inputs.append(y)

        if self.max_num_classes >= 2:
            y = self._multiclass_encode(inputs[0], single_eval_pos)
            inputs[0] = y

        x = torch.cat(inputs, dim=-1)
        return self.proj(x)
