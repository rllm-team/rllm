from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from torch import Tensor

from rllm.types import ColType, StatType
from ._col_encoder import ColEncoder


class RemoveConstantFeaturesEncoder(ColEncoder):
    r"""Remove features that are constant across rows.

    Unlike :class:`RemoveEmptyFeaturesEncoder`, this encoder returns a
    variable-width output and is intended for pipelines that explicitly
    repack features afterwards, such as TabPFN feature grouping.
    """

    supported_types = {
        ColType.NUMERICAL,
        ColType.CATEGORICAL,
        ColType.BINARY,
    }

    def __init__(
        self,
        out_dim: Optional[int] = None,
        stats_list: Optional[List[Dict[StatType, Any]]] = None,
        post_module: Optional[torch.nn.Module] = None,
    ) -> None:
        super().__init__(
            out_dim=out_dim,
            stats_list=stats_list,
            post_module=post_module,
            preserve_invalid_values=True,
        )

    def post_init(self) -> None:
        return None

    def reset_parameters(self) -> None:
        super().reset_parameters()

    def encode_forward(self, feat: Tensor, **kwargs: object) -> Tensor:
        del kwargs
        if feat.ndim not in (2, 3):
            raise ValueError(
                f"Expected feat to be 2D or 3D, but got shape {tuple(feat.shape)}."
            )
        if feat.shape[0] <= 1:
            out = feat
        else:
            selection_mask = ~(feat[1:] == feat[0]).all(0)
            if selection_mask.ndim > 1:
                selection_mask = selection_mask.any(0)
            out = feat[..., selection_mask.to(torch.bool)]

        return out
