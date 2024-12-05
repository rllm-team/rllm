from __future__ import annotations
from typing import Any, Dict, List

from torch import Tensor
from torch.nn import Module

from rllm.types import ColType, NAMode, StatType
from .coltype_encoder import ColTypeEncoder


class NumericalDefaultEncoder(ColTypeEncoder):
    r"""Simply fill na value in numerical features.
    :obj:`[batch_size, num_cols]` into
    :obj:`[batch_size, num_cols]`.
    """

    supported_types = {ColType.NUMERICAL}

    def __init__(
        self,
        out_dim: int | None = 1,
        col_type: ColType | None = ColType.NUMERICAL,
        post_module: Module | None = None,
        na_mode: NAMode | None = None,
    ) -> None:
        super().__init__(out_dim, col_type, post_module, na_mode)

    def post_init(self) -> None:
        pass

    def reset_parameters(self) -> None:
        pass

    def encode_forward(
        self,
        feat: Tensor,
    ) -> Tensor:
        # feat: [batch_size, num_cols]
        x = feat
        return x


class CategoricalDefaultEncoder(ColTypeEncoder):
    r"""Simply fill na value in categorical features.
    :obj:`[batch_size, num_cols]` into
    :obj:`[batch_size, num_cols]`.
    """

    supported_types = {ColType.CATEGORICAL}

    def __init__(
        self,
        out_dim: int | None = 1,
        stats_list: List[Dict[StatType, Any]] | None = None,
        col_type: ColType | None = ColType.CATEGORICAL,
        post_module: Module | None = None,
        na_mode: NAMode | None = None,
    ) -> None:
        super().__init__(out_dim, stats_list, col_type, post_module, na_mode)

    def post_init(self) -> None:
        pass

    def reset_parameters(self) -> None:
        pass

    def encode_forward(
        self,
        feat: Tensor,
    ) -> Tensor:
        # feat: [batch_size, num_cols]
        x = feat
        return x
