from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, List, Callable

import torch
from torch import Tensor

from rllm.data import TableData
from rllm.types import ColType, NAMode, StatType


def _reset_parameters_soft(module: torch.nn.Module):
    r"""Call reset_parameters() only when it exists. Skip activation module."""
    if hasattr(module, "reset_parameters") and callable(module.reset_parameters):
        module.reset_parameters()


def _get_na_mask(tensor: Tensor) -> Tensor:
    r"""Obtains the Na mask of the input :obj:`Tensor`.

    Args:
        tensor (Tensor): Input :obj:`Tensor`.
    """
    if tensor.is_floating_point():
        na_mask = torch.isnan(tensor)
    else:
        na_mask = tensor == -1
    return na_mask


class TableTransform(torch.nn.Module, ABC):
    r"""Base class for table Transform. This module transforms tensor of some
    specific columns type into 3-dimensional column-wise tensor that is input
    into tabular deep learning models. Columns with same ColType will be
    transformed into tensors. By default, it handles missing values (NaNs)
    according to the specified `na_mode`.

    Args:
        out_dim (int): The output dim dimensionality
        col_type (stype): The stype of the Transform input.
        post_module (Module, optional): The post-hoc module applied to the
            output, such as activation function and normalization. Must
            preserve the shape of the output. If :obj:`None`, no module will be
            applied to the output. (default: :obj:`None`)
        na_mode (NAMode, optional): The instruction that indicates how to
            impute NaN values. (default: :obj:`None`)
        transforms (List[Callable], optional): A list of transformation
            functions to be applied to the input data. Each function in the
            list should take the input data as an argument and return the
            transformed data. (default: :obj=`None`)
    """

    def __init__(
        self,
        out_dim: int | None = None,
        col_type: ColType | None = None,
        post_module: torch.nn.Module | None = None,
        na_mode: Dict[StatType, NAMode] | None = None,
        transforms: List[Callable] | None = None,
    ):
        r"""Since many attributes are specified later,
        this is a fake initialization"""
        super().__init__()

        if na_mode is not None:
            if (
                col_type == ColType.NUMERICAL
                and na_mode not in NAMode.namode_for_col_type(ColType.NUMERICAL)
            ):
                raise ValueError(f"{na_mode} cannot be used on numerical columns.")
            if (
                col_type == ColType.CATEGORICAL
                and na_mode not in NAMode.namode_for_col_type(ColType.CATEGORICAL)
            ):
                raise ValueError(f"{na_mode} cannot be used on categorical columns.")
        else:
            na_mode = {
                ColType.NUMERICAL: NAMode.MEAN,
                ColType.CATEGORICAL: NAMode.MOST_FREQUENT,
            }

        self.out_dim = out_dim
        self.post_module = post_module
        self.na_mode = na_mode
        self.transforms = transforms

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
        data: TableData,
    ) -> Tensor:
        # NaN handling of the input Tensor
        data = self.nan_forward(data)

        for transform in self.transforms:
            data = transform(data)

        return data

    def nan_forward(
        self,
        data: TableData,
    ) -> Tensor:
        r"""Replace NaN values in input :obj:`Tensor` given
        :obj:`na_mode`.

        Args:
            feat: Input :obj:`Tensor`.

        Returns:
            Tensor: Output :obj:`Tensor` with NaNs replaced given
            :obj:`na_mode`.
        """
        if self.na_mode is None:
            return data

        # Since we are not changing the number of items in each column, it's
        # faster to just clone the values, while reusing the same offset
        # object.
        feats = data.get_feat_dict()
        for col_type, feat in feats.items():
            feat = self._fill_nan(feat, data.metadata[col_type], self.na_mode[col_type])
            # Handle NaN in case na_mode is None
            feats[col_type] = torch.nan_to_num(feat, nan=0)

        data.feat_dict = feats
        return data

    def _fill_nan(
        self,
        feat: Tensor,
        stats_list: Dict[StatType, float],
        na_mode: NAMode,
    ) -> Tensor:
        r"""Replace NaN values in input :obj:`Tensor` given :obj:`na_mode`."""
        if isinstance(feat, Tensor):
            # cache for future use
            na_mask = _get_na_mask(feat)
            if na_mask.any():
                feat = feat.clone()
            else:
                return feat
        else:
            raise ValueError(f"Unrecognized type {type(feat)} in na_forward.")

        fill_values = []
        for col in range(feat.size(1)):
            if na_mode == NAMode.MOST_FREQUENT:
                fill_value = stats_list[col][StatType.MOST_FREQUENT]
            elif na_mode == NAMode.MEAN:
                fill_value = stats_list[col][StatType.MEAN]
            elif na_mode == NAMode.ZERO:
                fill_value = 0
            else:
                raise ValueError(f"Unsupported NA mode {self.na_mode}")
            fill_values.append(fill_value)

        if na_mask.ndim == 3:
            # when feat is 3D, it is faster to iterate over columns
            for col, fill_value in enumerate(fill_values):
                col_data = feat[:, col]
                col_na_mask = na_mask[:, col].any(dim=-1)
                col_data[col_na_mask] = fill_value
        else:  # na_mask.ndim == 2
            fill_values = torch.tensor(fill_values, device=feat.device)
            assert feat.size(-1) == fill_values.size(-1)
            feat = torch.where(na_mask, fill_values, feat)

        # Add better safeguard here to make sure nans are actually
        # replaced, expecially when nans are represented as -1's. They are
        # very hard to catch as they won't error out.
        filled_values = feat

        if filled_values.is_floating_point():
            assert not torch.isnan(filled_values).any()
        else:
            assert not (filled_values == -1).any()
        return feat
