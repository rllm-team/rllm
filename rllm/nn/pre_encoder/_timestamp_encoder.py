from __future__ import annotations
from typing import Any, Optional, Dict, List

import pandas as pd
import torch
from torch import Tensor
from torch.nn import Parameter

from rllm.types import ColType, StatType
from rllm.preprocessing import TimestampPreprocessor
from ._col_encoder import ColEncoder
from .positional_encoder import PositionalEncoder
from .cyclic_encoder import CyclicEncoder


class TimestampEncoder(ColEncoder):
    r"""TimestampEncoder for TIMESTAMP ColType.
    The NA value will be fulfilled by MEDIAN_TIMESTAMP.

    Args:
        out_dim (int, optional): The output dimensionality (default: :obj:`1`).
        stats_list (List[Dict[rllm.types.StatType, Any]], optional): The list of statistics
            for each column within the same column type (default: :obj:`None`).
        post_module (torch.nn.Module, optional): The post-hoc module applied to the
            output, such as activation function and normalization. Must
            preserve the shape of the output. If :obj:`None`, no module will
            be applied to the output (default: :obj:`None`).
        out_size (int, optional): Output dimension of the positional and cyclic
            encodings. (default: :obj:`8`).
        fill_nan (bool, optional): Whether to fill NaN values with the median timestamp.
            If True, the median timestamp will be used to fill in the NaN values.
            (default: :obj:`False`).
    """

    supported_types = {ColType.TIMESTAMP}

    def __init__(
        self,
        out_dim: Optional[int] = None,
        stats_list: Optional[List[Dict[StatType, Any]]] = None,
        post_module: Optional[torch.nn.Module] = None,
        out_size: int = 8,
        fill_nan: bool = False,
    ) -> None:
        self.out_size = out_size
        self.fill_nan = fill_nan
        super().__init__(out_dim, stats_list, post_module)

    def post_init(self) -> None:

        # Init normalization constant
        if self.fill_nan:
            median_time = pd.to_datetime(
                pd.Series(
                    [
                        self.stats_list[i][StatType.MEDIAN_TIME]
                        for i in range(len(self.stats_list))
                    ]
                )
            )
            median_time = TimestampPreprocessor.to_tensor(
                median_time
            )  # [Col, num_time_feats]
            self.register_buffer("median_time_tensor", median_time)

        min_year = torch.tensor(
            [
                self.stats_list[i][StatType.YEAR_RANGE][0]
                for i in range(len(self.stats_list))
            ]
        )
        self.register_buffer("min_year", min_year)
        max_values = TimestampPreprocessor.CYCLIC_VALUES_NORMALIZATION_CONSTANT
        self.register_buffer("max_values", max_values)

        # Init positional/cyclic encoding
        self.positional_encoding = PositionalEncoder(self.out_size)
        self.cyclic_encoding = CyclicEncoder(self.out_size)

        # Init linear function
        num_cols = len(self.stats_list)
        self.weight = Parameter(
            torch.empty(
                num_cols,
                len(TimestampPreprocessor.TIME_TO_INDEX),
                self.out_size,
                self.out_dim,
            )
        )
        self.bias = Parameter(torch.empty(num_cols, self.out_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        super().reset_parameters()
        torch.nn.init.normal_(self.weight, std=0.01)
        torch.nn.init.zeros_(self.bias)

    def encode_forward(self, feat: Tensor) -> Tensor:
        if self.fill_nan:
            # fill in NaN with median timestamp
            fill_values = self.median_time_tensor.unsqueeze(0).expand_as(feat)
            na_mask = feat[..., 0] == -1  # only check if year is NaN, [B, num_cols]
            na_mask = na_mask.unsqueeze(-1).expand_as(feat)
            feat[na_mask] = fill_values[na_mask]

        feat = feat.to(torch.float32)
        # [batch_size, num_cols, 1] - [1, num_cols, 1]
        feat_year = feat[..., :1] - self.min_year.view(1, -1, 1)
        # [batch_size, num_cols, num_rest] / [1, 1, num_rest]
        feat_rest = feat[..., 1:] / self.max_values.view(1, 1, -1)
        # [batch_size, num_cols, num_time_feats, out_size]
        x = torch.cat(
            [self.positional_encoding(feat_year), self.cyclic_encoding(feat_rest)],
            dim=2,
        )
        # [batch_size, num_cols, num_time_feats, out_size] *
        # [num_cols, num_time_feats, out_size, out_channels]
        # -> [batch_size, num_cols, out_channels]
        x_lin = torch.einsum("ijkl,jklm->ijm", x, self.weight)
        # [batch_size, num_cols, out_channels] + [num_cols, out_channels]
        x = x_lin + self.bias
        return x
