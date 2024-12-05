from __future__ import annotations
from typing import Dict, List, Callable
from abc import ABC, abstractmethod

import torch
from torch import Tensor
from torch.nn import (
    Module,
    Sequential,
)

from rllm.data import TableData
from rllm.types import ColType, NAMode, StatType


def _reset_parameters_soft(module: Module):
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


class TableTransform(Module, ABC):
    r"""Base class for columns Transform. This module transforms tensor of some
    specific columns type into 3-dimensional column-wise tensor
    that is input into tabular deep learning models.
    Columns with same ColType will be transformed into tensors.

    Args:
        out_dim (int): The output dim dimensionality
        stats_list (List[Dict[StatType]]): The list
            of stats for each column within the same column type.
        col_type (stype): The stype of the Transform input.
        post_module (Module, optional): The post-hoc module applied to the
            output, such as activation function and normalization. Must
            preserve the shape of the output. If :obj:`None`, no module will be
            applied to the output. (default: :obj:`None`)
        na_mode (NAMode, optional): The instruction that indicates how to
            impute NaN values. (default: :obj:`None`)
    """

    def __init__(
        self,
        out_dim: int | None = None,
        stats_list: List[Dict[StatType]] | None = None,
        col_type: ColType | None = None,
        post_module: Module | None = None,
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

        self.out_dim = out_dim
        self.stats_list = stats_list
        self.post_module = post_module
        self.na_mode = na_mode
        self.transforms = transforms

    @abstractmethod
    def reset_parameters(self):
        r"""Initialize the parameters of `post_module`."""
        if self.post_module is not None:
            if isinstance(self.post_module, Sequential):
                for m in self.post_module:
                    _reset_parameters_soft(m)
            else:
                _reset_parameters_soft(self.post_module)

    def forward(
        self,
        data: TableData,
    ) -> Tensor:
        # NaN handling of the input Tensor
        data = self.na_forward(data)
        # Main encoding into column embeddings
        data = self.encode_forward(data)
        # Post-forward (e.g., normalization, activation)
        return self.post_forward(data)

    def encode_forward(
        self,
        data: TableData,
    ) -> Tensor:
        r"""The main forward function. Maps input :obj:`feat` from feat_dict
        (shape [batch_size, num_cols]) into output :obj:`x` of shape
        :obj:`[batch_size, num_cols, out_dim]`.
        """
        for transform in self.transforms:
            data = transform(data)
        return data

    def post_forward(self, out: TableData) -> Tensor:
        r"""Post-forward function applied to :obj:`out` of shape
        [batch_size, num_cols, dim]. It also returns :obj:`out` of the
        same shape.
        """
        if self.post_module is not None:
            feats = out.get_feat_dict()
            for col_type, feat in feats.items():
                shape_before = feat.shape
                feat = self.post_module(feat)
                if feat.shape != shape_before:
                    raise RuntimeError(
                        f"post_module must not alter the shape of the tensor, but "
                        f"it changed the shape from {shape_before} to "
                        f"{feat.shape}."
                    )
                feats[col_type] = feat

        return out

    def na_forward(
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
            feat = self._fill_nan(feat, self.na_mode[col_type])
            # Handle NaN in case na_mode is None
            feats[col_type] = torch.nan_to_num(feat, nan=0)

        return feats

    def _fill_nan(
        self,
        feat: Tensor,
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
                fill_value = self.stats_list[col][StatType.MOST_FREQUENT]
            elif na_mode == NAMode.MEAN:
                fill_value = self.stats_list[col][StatType.MEAN]
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

    def _nan_to_num(self, x: Tensor) -> Tensor:
        r"""Replace NaN values in input :obj:`Tensor` with 0."""
        if x.is_floating_point():
            return torch.nan_to_num(x, nan=0)
        else:
            return torch.where(x == -1, torch.tensor(0, device=x.device), x)
