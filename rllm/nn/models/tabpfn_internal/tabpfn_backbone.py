from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from .tabpfn_transformer import TabPFNBlock


@dataclass
class TabPFNConfig:
    """Configuration for the TabPFN transformer backbone."""

    emsize: int = 192
    nlayers: int = 24
    nhead: int = 3
    features_per_group: int = 3
    num_thinking_rows: int = 64
    encoder_type: Literal["linear", "mlp"] = "linear"
    encoder_mlp_hidden_dim: int = 1024


class TabPFNBackbone(nn.Module):
    """Pure transformer backbone used by the TabPFN model."""

    def __init__(
        self,
        *,
        config: TabPFNConfig,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.input_size = int(config.emsize)
        self.hidden_size = self.input_size * 2
        self.blocks = nn.ModuleList(
            TabPFNBlock(
                emsize=self.input_size,
                nhead=config.nhead,
                dim_feedforward=self.hidden_size,
                device=device,
                dtype=dtype,
            )
            for _ in range(config.nlayers)
        )

    def forward(
        self,
        x_BRCE: torch.Tensor,
        *,
        single_eval_pos: int,
        save_peak_memory_factor: int | None = None,
    ) -> torch.Tensor:
        for block in self.blocks:
            x_BRCE = block(
                x_BRCE,
                single_eval_pos=single_eval_pos,
                save_peak_memory_factor=save_peak_memory_factor,
            )
        return x_BRCE
