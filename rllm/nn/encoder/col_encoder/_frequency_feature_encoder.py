from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from torch import Tensor

from rllm.types import ColType, StatType
from ._col_encoder import ColEncoder


class FrequencyFeatureEncoder(ColEncoder):
    r"""Expand each numerical column with sinusoidal frequency features."""

    supported_types = {ColType.NUMERICAL}

    def __init__(
        self,
        num_frequencies: int,
        freq_power_base: float = 2.0,
        max_wave_length: float = 4.0,
        out_dim: Optional[int] = None,
        stats_list: Optional[List[Dict[StatType, Any]]] = None,
        post_module: Optional[torch.nn.Module] = None,
    ) -> None:
        self.num_frequencies = num_frequencies
        self.freq_power_base = freq_power_base
        computed_out_dim = 1 + 2 * num_frequencies
        super().__init__(
            out_dim=computed_out_dim if out_dim is None else out_dim,
            stats_list=stats_list,
            post_module=post_module,
        )

        wave_lengths = torch.tensor(
            [freq_power_base**i for i in range(num_frequencies)],
            dtype=torch.float,
        )
        wave_lengths = wave_lengths / wave_lengths[-1] * max_wave_length
        self.register_buffer("wave_lengths", wave_lengths)

    def post_init(self) -> None:
        return None

    def reset_parameters(self) -> None:
        super().reset_parameters()

    def encode_forward(self, feat: Tensor) -> Tensor:
        if feat.ndim == 2:
            x = feat.unsqueeze(-1)
        elif feat.ndim == 3:
            x = feat
            if x.shape[-1] != 1:
                raise ValueError(
                    "FrequencyFeatureEncoder expects last dim == 1 for 3D input."
                )
        else:
            raise ValueError(
                f"Expected feat to be 2D or 3D, but got shape {tuple(feat.shape)}."
            )

        extended = x[..., None] / self.wave_lengths[None, None, None, :] * 2 * torch.pi
        out = torch.cat(
            (x[..., None], torch.sin(extended), torch.cos(extended)), dim=-1
        )
        out = out.reshape(*x.shape[:-1], -1)

        if self.post_module is not None:
            out = self.post_module(out)

        return out
