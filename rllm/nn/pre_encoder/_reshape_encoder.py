from __future__ import annotations
from typing import Any, Dict, List

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
    """

    supported_types = {ColType.CATEGORICAL, ColType.NUMERICAL}

    def __init__(
        self,
        out_dim: int | None = 1,
        stats_list: List[Dict[StatType, Any]] | None = None,
        post_module: torch.nn.Module | None = None,
    ) -> None:
        super().__init__(out_dim, stats_list, post_module)

    def post_init(self) -> None:
        pass

    def reset_parameters(self) -> None:
        pass

    def encode_forward(
        self,
        feat: Tensor,
    ) -> Tensor:
        # feat: [batch_size, num_cols]
        if feat.dim() != 3:
            feat = feat.unsqueeze(2)

        return feat
