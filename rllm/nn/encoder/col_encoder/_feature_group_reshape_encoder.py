from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from torch import Tensor

from rllm.types import ColType, StatType
from ._col_encoder import ColEncoder


class FeatureGroupReshapeEncoder(ColEncoder):
    r"""Pad and reshape flat features into fixed-size feature groups."""

    supported_types = {ColType.NUMERICAL}

    def __init__(
        self,
        num_features_per_group: int,
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
        self.num_features_per_group = int(num_features_per_group)
        self.num_feature_groups: int = 0

    def post_init(self) -> None:
        return None

    def reset_parameters(self) -> None:
        super().reset_parameters()

    def encode_forward(self, feat: Tensor, **kwargs: object) -> Tensor:
        del kwargs
        if feat.ndim != 3:
            raise ValueError(
                f"Expected feat to be 3D, but got shape {tuple(feat.shape)}."
            )

        num_columns = feat.shape[-1]
        num_padding_features = -num_columns % self.num_features_per_group
        feat_padded = torch.nn.functional.pad(
            feat,
            pad=(0, num_padding_features),
            value=0,
        )
        num_rows, batch_size, num_padded_columns = feat_padded.shape
        self.num_feature_groups = num_padded_columns // self.num_features_per_group
        out = feat_padded.reshape(
            num_rows,
            batch_size * self.num_feature_groups,
            self.num_features_per_group,
        )
        return out
