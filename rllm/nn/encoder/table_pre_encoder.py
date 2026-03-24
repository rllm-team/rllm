from __future__ import annotations
from typing import Any, Dict, List, Union
from abc import ABC

import torch
from torch import Tensor

from .col_encoder._col_encoder import ColEncoder
from rllm.types import ColType


class TablePreEncoder(torch.nn.Module, ABC):
    r"""The TablePreEncoder class is designed to transform table data by encoding
    each column type tensor into embeddings and performing the final
    concatenation. It supports different types of column encoders for
    categorical and numerical features, allowing for flexible and
    efficient preprocessing of tabular data.

    Args:
        out_dim (int): Output dimensionality.
        metadata (Dict[ColType, List[Dict[str, Any]]]): Metadata for each column
            type, specifying the statistics and properties of the columns.
        col_encoder_dict
            (Dict[:class:`rllm.types.ColType`,
            :class:`rllm.nn.encoder.ColEncoder]):
            A dictionary that maps :class:`rllm.types.ColType` into
            :class:`rllm.nn.encoder.ColEncoder` class. Only
            parent :class:`stypes <rllm.types.ColType>` are supported
            as keys.

    Example:
        >>> from rllm.nn.encoder import TablePreEncoder
        >>> # Usually instantiated through concrete subclasses.
    """

    def __init__(
        self,
        out_dim: int,
        metadata: Dict[ColType, List[Dict[str, Any]]],
        col_encoder_dict: Dict[ColType, List[ColEncoder]],
    ) -> None:
        super().__init__()

        self.metadata = metadata
        self.col_encoder_dict = torch.nn.ModuleDict()

        for col_type, col_encoders in col_encoder_dict.items():
            encoder_list = torch.nn.ModuleList()
            for col_encoder in col_encoders:
                if col_type not in col_encoder.supported_types:
                    raise ValueError(
                        f"{col_encoder} does not " f"support encoding {col_type}."
                    )
                # Set attributes
                if col_encoder.out_dim is None:
                    col_encoder.out_dim = out_dim
                if col_type in metadata.keys():
                    col_encoder.stats_list = metadata[col_type]
                    col_encoder.post_init()
                    encoder_list.append(col_encoder)

            if len(encoder_list) > 0:
                self.col_encoder_dict[col_type.value] = encoder_list
        self.reset_parameters()

    def reset_parameters(self):
        """Reset parameters for all encoders in the encoder_dict."""
        for col_encoder_list in self.col_encoder_dict.values():
            for col_encoder in col_encoder_list:
                col_encoder.reset_parameters()

    def forward(
        self,
        feat_dict: Dict[ColType, Tensor],
        return_dict: bool = False,
    ) -> Union[Tensor, Dict[ColType, Tensor]]:
        r"""Encode each column type and concatenate the results.

        Args:
            feat_dict (Dict[ColType, Tensor]): Input features grouped by
                column type.
            return_dict (bool): If :obj:`True`, return a dictionary of
                per-column-type embeddings instead of a concatenated tensor.
                (default: :obj:`False`)

        Returns:
            Union[Tensor, Dict[ColType, Tensor]]: Concatenated embedding
            tensor of shape :obj:`[batch_size, num_cols, out_dim]`, or a
            dictionary of per-column-type embeddings when
            :obj:`return_dict=True`.
        """
        feat_encoded = {}
        for col_type in feat_dict.keys():
            feat = feat_dict[col_type]
            if col_type.value in self.col_encoder_dict.keys():
                x = feat
                for col_encoder in self.col_encoder_dict[col_type.value]:
                    x = col_encoder(x)
                feat_encoded[col_type] = x
            else:
                feat_encoded[col_type] = feat

        if return_dict:
            return feat_encoded

        feat_list = list(feat_encoded.values())
        return torch.cat(feat_list, dim=1)
