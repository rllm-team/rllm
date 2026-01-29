from __future__ import annotations
from typing import Any, Dict, List, Optional

import torch
from torch import Tensor

from ._col_encoder import ColEncoder
from rllm.types import ColType, StatType


class ReshapeEncoder(ColEncoder):
    r"""A simple reshaping-based ColEncoder for categorical and numerical features.
    This encoder reshapes the input tensor from :obj:`[batch_size, num_cols]`
    to :obj:`[batch_size, num_cols, 1]` if it is not already 3-dimensional.
    It is designed to handle both categorical and numerical features.

    Args:
        out_dim (int, optional): The output dimensionality (default: :obj:`1`).
        stats_list (List[Dict[StatType, Any]], optional): The list of statistics
            for each column within the same column type (default: :obj:`None`).
        post_module (torch.nn.Module, optional): The post-hoc module applied
            to the output, such as activation function and normalization. Must
            preserve the shape of the output. If :obj:`None`, no module will
            be applied to the output (default: :obj:`None`).
        need_layer_norm (bool, optional): Whether to apply LayerNorm to the input.
    """

    supported_types = {ColType.CATEGORICAL, ColType.NUMERICAL}

    def __init__(
        self,
        out_dim: int = 1,
        stats_list: List[Dict[StatType, Any]] = None,
        post_module: torch.nn.Module = None,
        need_layer_norm: bool = True,
    ) -> None:
        self.need_layer_norm = need_layer_norm
        super().__init__(out_dim, stats_list, post_module)

    def post_init(self) -> None:
        if self.need_layer_norm:
            self.layernorm = torch.nn.LayerNorm(len(self.stats_list))

    def reset_parameters(self) -> None:
        self.layernorm.reset_parameters()

    def encode_forward(
        self,
        feat: Tensor,
    ) -> Tensor:
        # feat: [batch_size, num_cols]
        if self.need_layer_norm:
            feat = self.layernorm(feat)
        if feat.dim() != 3:
            feat = feat.unsqueeze(2)

        return feat
