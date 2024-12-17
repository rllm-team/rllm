from __future__ import annotations
from typing import Any, Dict, List

from torch import Tensor
from torch.nn import Module

from ._col_encoder import ColEncoder
from rllm.types import ColType, StatType


class ReshapeEncoder(ColEncoder):
    r"""Simply fill na value in categorical features.
    :obj:`[batch_size, num_cols]` into
    :obj:`[batch_size, num_cols]`.
    """

    supported_types = {ColType.CATEGORICAL, ColType.NUMERICAL}

    def __init__(
        self,
        out_dim: int | None = 1,
        stats_list: List[Dict[StatType, Any]] | None = None,
        post_module: Module | None = None,
    ) -> None:
        super().__init__(
            out_dim,
            stats_list,
            post_module,
        )

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
