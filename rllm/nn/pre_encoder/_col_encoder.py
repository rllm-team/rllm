from __future__ import annotations

from typing import Dict, List
from abc import ABC, abstractmethod

import torch
from torch import Tensor

from rllm.types import ColType, StatType


def _reset_parameters_soft(module: torch.nn.Module):
    r"""Call reset_parameters() only when it exists. Skip activation module."""
    if hasattr(module, "reset_parameters") and callable(module.reset_parameters):
        module.reset_parameters()


class ColEncoder(torch.nn.Module, ABC):
    r"""Base class for columns pre_encoder. This module encodes tensor of some
    specific columns type into 3-dimensional column-wise tensor
    that is input into tabular deep learning models.
    Columns with same ColType will be encoded into tensors.

    Args:
        out_dim (int): The output dim dimensionality
        stats_list (List[Dict[StatType]]): The list
            of stats for each column within the same column type.
        post_module (torch.nn.Module, optional): The post-hoc module applied to the
            output, such as activation function and normalization. Must
            preserve the shape of the output. If :obj:`None`, no module will be
            applied to the output. (default: :obj:`None`)
    """

    supported_types: set[ColType] = {}

    def __init__(
        self,
        out_dim: int | None = None,
        stats_list: List[Dict[StatType]] | None = None,
        post_module: torch.nn.Module | None = None,
    ):
        r"""Since many attributes are specified later,
        this is a fake initialization"""
        super().__init__()

        self.out_dim = out_dim
        self.stats_list = stats_list
        self.post_module = post_module

    @abstractmethod
    def post_init(self):
        raise NotImplementedError

    @abstractmethod
    def reset_parameters(self):
        r"""Initialize the parameters of `post_module`."""
        if self.post_module is not None:
            if isinstance(self.post_module, torch.nn.Sequential):
                for m in self.post_module:
                    _reset_parameters_soft(m)
            else:
                _reset_parameters_soft(self.post_module)

    def forward(
        self,
        feat: Tensor,
        col_names: List[str] | None = None,
    ) -> Tensor:
        if col_names is not None:
            num_cols = feat.shape[1]
            if num_cols != len(col_names):
                raise ValueError(
                    f"The number of columns in feat and the length of "
                    f"col_names must match (got {num_cols} and "
                    f"{len(col_names)}, respectively.)"
                )

        # Main encoding into column embeddings
        x = self.encode_forward(feat)
        # Handle NaN in case na_mode is None
        x = torch.nan_to_num(x, nan=0)
        return x

    @abstractmethod
    def encode_forward(
        self,
        feat: Tensor,
    ) -> Tensor:
        r"""The main forward function. Maps input :obj:`feat` from feat_dict
        (shape [batch_size, num_cols]) into output :obj:`x` of shape
        :obj:`[batch_size, num_cols, out_dim]`.
        """
        raise NotImplementedError
