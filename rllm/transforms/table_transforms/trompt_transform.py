from __future__ import annotations
from typing import Any, Dict, List, Tuple

from torch import Tensor
import torch

from rllm.types import ColType
from rllm.transforms.table_transforms import (
    ColTypeTransform,
    TableTypeTransform,
)
from rllm.nn.pre_encoder import EmbeddingEncoder, LinearEncoder


class TromptTransform(TableTypeTransform):
    def __init__(
        self,
        out_dim: int = None,
        metadata: Dict[ColType, List[Dict[str, Any]]] = None,
        col_types_transform_dict: Dict[ColType, ColTypeTransform] = None,
    ) -> None:
        if col_types_transform_dict is None:
            col_types_transform_dict = {
                ColType.CATEGORICAL: EmbeddingEncoder(),
                ColType.NUMERICAL: LinearEncoder(activate=torch.nn.ReLU()),
            }
        self._initialized = False
        self.out_dim = out_dim
        self.metadata = metadata
        self.col_types_transform_dict = col_types_transform_dict
        self.layer_norm = torch.nn.LayerNorm(out_dim)

    def __setattr__(self, name, value):
        # Hacky way to delay initialization
        super().__setattr__(name, value)

        if not self._initialized and all(
            [
                hasattr(self, "out_dim") and self.out_dim,
                hasattr(self, "metadata") and self.metadata,
                hasattr(self, "col_types_transform_dict"),
            ]
        ):
            self._initialized = True
            super().__init__(self.out_dim, self.metadata, self.col_types_transform_dict)

    def forward(self, feat_dict: Dict[ColType, Tensor]) -> Tuple[Tensor, List[str]]:
        return self.layer_norm(super().forward(feat_dict))
