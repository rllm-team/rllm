from __future__ import annotations
from typing import Any, Dict, List, Tuple
from abc import ABC

import torch
from torch import Tensor
from torch.nn import Module, ModuleDict

from rllm.types import ColType
from .coltype_encoder import ColTypeEncoder


class PreEncoder(Module, ABC):
    r"""Table Transform that transforms each ColType tensor into embeddings and
    performs the final concatenation.

    Args:
        out_dim (int): Output dimensionality.
        metadata
            (Dict[class:`rllm.types.ColType`, List[dict[StatType]]):
            A dictionary that maps column type into stats.
        col_types_encoder_dict
            (Dict[:class:`rllm.types.ColType`,
            :class:`rllm.nn.encoder.ColTypeTransform`]):
            A dictionary that maps :class:`rllm.types.ColType` into
            :class:`rllm.nn.encoder.ColTypeTransform` class. Only
            parent :class:`stypes <rllm.types.ColType>` are supported
            as keys.
    """

    def __init__(
        self,
        out_dim: int,
        metadata: Dict[ColType, List[Dict[str, Any]]],
        col_types_encoder_dict: Dict[ColType, ColTypeEncoder],
    ) -> None:
        super().__init__()

        # self.metadata = metadata
        self.transform_dict = ModuleDict()

        for col_type, col_types_transform in col_types_encoder_dict.items():
            if col_type not in col_types_transform.supported_types:
                raise ValueError(
                    f"{col_types_transform} does not " f"support encoding {col_type}."
                )
            # Set attributes
            col_types_transform.col_type = col_type
            if col_types_transform.out_dim is None:
                col_types_transform.out_dim = out_dim
            if col_type in metadata.keys():
                col_types_transform.stats_list = metadata[col_type]
            self.transform_dict[col_type.value] = col_types_transform
            col_types_transform.post_init()
        self.reset_parameters()

    def reset_parameters(self):
        """Reset parameters for all transforms in the transform_dict."""
        for transform in self.transform_dict.values():
            transform.reset_parameters()

    def forward(
        self,
        feat_dict: Dict[ColType, Tensor],
    ) -> Tuple[Tensor, List[str]]:
        xs = []
        for col_type in feat_dict.keys():
            feat = feat_dict[col_type]
            if col_type.value in self.transform_dict.keys():
                x = self.transform_dict[col_type.value](feat)
                xs.append(x)
        x = torch.cat(xs, dim=1)
        return x
