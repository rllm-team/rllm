from __future__ import annotations
from typing import Any, Dict, List
import math

import torch

from ._col_encoder import ColEncoder
from rllm.types import ColType, StatType


class TransTabNumEmbeddingEncoder(ColEncoder):
    r"""Numerical feature encoder for TransTab.

    Encodes each numerical column by multiplying a learned embedding vector
    by the raw value and adding a per-feature bias. Adapted from the original
    TransTabNumEmbedding implementation.

    Args:
        hidden_dim (int): Dimensionality of the output embedding for each feature.
        stats_list (Optional[List[Dict[StatType, Any]]]): Precomputed statistics
            (e.g., mean, std) for each numerical column. (default: None)
        post_module (Optional[torch.nn.Module]): Optional module to apply
            after encoding (e.g., batch normalization or activation). (default: None)
    """

    supported_types = {ColType.NUMERICAL}

    def __init__(
        self,
        hidden_dim: int,
        stats_list: List[Dict[StatType, Any]] | None = None,
        post_module: torch.nn.Module | None = None,
    ) -> None:
        super().__init__(out_dim=hidden_dim, stats_list=stats_list, post_module=post_module)
        self.out_dim = hidden_dim
        self.num_bias = torch.nn.Parameter(torch.empty(1, 1, hidden_dim))
        self.reset_parameters()

    def reset_parameters(self):
        # Reset parent class and bias parameters
        super().reset_parameters()
        torch.nn.init.uniform_(
            self.num_bias,
            a=-1 / math.sqrt(self.out_dim),
            b=1 / math.sqrt(self.out_dim),
        )

    def encode_forward(
        self,
        feat: torch.Tensor,         # [num_cols, hidden_dim]
        col_names: List[str] | None,
        raw_vals: torch.Tensor,     # [batch, num_cols]
    ) -> torch.Tensor:
        num_col_emb = feat.unsqueeze(0).expand(raw_vals.shape[0], -1, -1)
        return num_col_emb * raw_vals.unsqueeze(-1).float() + self.num_bias

    def post_init(self):
        return

    def forward(
        self,
        feat: torch.Tensor,
        col_names: List[str] | None = None,
        raw_vals: torch.Tensor = None,
    ) -> torch.Tensor:
        return self.encode_forward(feat, col_names, raw_vals)
