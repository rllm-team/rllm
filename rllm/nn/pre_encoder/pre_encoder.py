from __future__ import annotations
from typing import Any, Dict, List, Tuple
from abc import ABC

import torch
from torch import Tensor
from torch.nn import Module, ModuleDict

from .coltype_encoder import ColTypeEncoder
from rllm.types import ColType


class PreEncoder(Module, ABC):
    r"""Table Transform that encoders each ColType tensor into embeddings and
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

        self.metadata = metadata
        self.encoder_dict = ModuleDict()

        for col_type, col_types_encoder in col_types_encoder_dict.items():
            if col_type not in col_types_encoder.supported_types:
                raise ValueError(
                    f"{col_types_encoder} does not " f"support encoding {col_type}."
                )
            # Set attributes
            if col_types_encoder.out_dim is None:
                col_types_encoder.out_dim = out_dim
            if col_type in metadata.keys():
                col_types_encoder.stats_list = metadata[col_type]
            self.encoder_dict[col_type.value] = col_types_encoder
            col_types_encoder.post_init()
        self.reset_parameters()

    def reset_parameters(self):
        """Reset parameters for all encoders in the encoder_dict."""
        for encoder in self.encoder_dict.values():
            encoder.reset_parameters()

    def forward(
        self,
        feat_dict: Dict[ColType, Tensor],
        return_dict: bool = False,
    ) -> Tuple[Tensor, List[str]]:
        feat_encoded = {}
        for col_type in feat_dict.keys():
            feat = feat_dict[col_type]
            if col_type.value in self.encoder_dict.keys():
                x = self.encoder_dict[col_type.value](feat)
                feat_encoded[col_type] = x
            else:
                feat_encoded[col_type] = feat

        if return_dict:
            return feat_encoded

        feat_list = list(feat_encoded.values())
        return torch.cat(feat_list, dim=1)
