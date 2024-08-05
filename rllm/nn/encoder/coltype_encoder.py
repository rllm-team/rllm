from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import Tensor
from torch.nn import (
    Embedding,
    Module,
    Parameter,
    Sequential,
)

from rllm.types import ColType, NAMode, StatType


def _reset_parameters_soft(module: Module):
    r"""Call reset_parameters() only when it exists. Skip activation module."""
    if hasattr(
        module, "reset_parameters"
    ) and callable(
        module.reset_parameters
    ):
        module.reset_parameters()


def _get_na_mask(tensor: Tensor) -> Tensor:
    r"""Obtains the Na mask of the input :obj:`Tensor`.

    Args:
        tensor (Tensor): Input :obj:`Tensor`.
    """
    if tensor.is_floating_point():
        na_mask = torch.isnan(tensor)
    else:
        na_mask = (tensor == -1)
    return na_mask


class ColTypeTransform(Module, ABC):
    r"""Base class for columns Transform. This module transforms tensor of some
    specific columns type into 3-dimensional column-wise tensor
    that is input into tabular deep learning models.
    Columns with same ColType will be transformed into tensors.

    Args:
        out_channels (int): The output channel dimensionality
        stats_list (list[dict[StatType, Any]]): The list
            of stats for each column within the same column type.
        col_type (stype): The stype of the Transform input.
        post_module (Module, optional): The post-hoc module applied to the
            output, such as activation function and normalization. Must
            preserve the shape of the output. If :obj:`None`, no module will be
            applied to the output. (default: :obj:`None`)
        na_mode (NAMode, optional): The instruction that indicates how to
            impute NaN values. (default: :obj:`None`)
    """

    supported_types: set[ColType] = {}

    def __init__(
        self,
        out_channels: int | None = None,
        stats_list: list[dict[StatType, Any]] | None = None,
        col_type: ColType | None = None,
        post_module: Module | None = None,
        na_mode: NAMode | None = None,
    ):
        r"""Since many attributes are specified later,
        this is a fake initialization"""
        super().__init__()

        if na_mode is not None:
            if (
                col_type == ColType.NUMERICAL
                and na_mode not in NAMode.namode_for_col_type(
                    ColType.NUMERICAL
                )
            ):
                raise ValueError(
                    f"{na_mode} cannot be used on numerical columns."
                )
            if (
                col_type == ColType.CATEGORICAL
                and na_mode not in NAMode.namode_for_col_type(
                    ColType.CATEGORICAL
                )
            ):
                raise ValueError(
                    f"{na_mode} cannot be used on categorical columns."
                )

        self.out_channels = out_channels
        self.stats_list = stats_list
        self.post_module = post_module
        self.na_mode = na_mode

    @abstractmethod
    def post_init(self):
        raise NotImplementedError

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
        feat: Tensor,
        col_names: list[str] | None = None,
    ) -> Tensor:
        if col_names is not None:
            num_cols = feat.shape[1]
            if num_cols != len(col_names):
                raise ValueError(
                    f"The number of columns in feat and the length of "
                    f"col_names must match (got {num_cols} and "
                    f"{len(col_names)}, respectively.)"
                )
        # NaN handling of the input Tensor
        feat = self.na_forward(feat)
        # Main encoding into column embeddings
        x = self.encode_forward(feat, col_names)
        # Handle NaN in case na_mode is None
        x = torch.nan_to_num(x, nan=0)
        # Post-forward (e.g., normalization, activation)
        return self.post_forward(x)

    @abstractmethod
    def encode_forward(
        self,
        feat: Tensor,
        col_names: list[str] | None = None,
    ) -> Tensor:
        r"""The main forward function. Maps input :obj:`feat` from feat_dict
        (shape [batch_size, num_cols]) into output :obj:`x` of shape
        :obj:`[batch_size, num_cols, out_channels]`.
        """
        raise NotImplementedError

    def post_forward(self, out: Tensor) -> Tensor:
        r"""Post-forward function applied to :obj:`out` of shape
        [batch_size, num_cols, channels]. It also returns :obj:`out` of the
        same shape.
        """
        if self.post_module is not None:
            shape_before = out.shape
            out = self.post_module(out)
            if out.shape != shape_before:
                raise RuntimeError(
                    f"post_module must not alter the shape of the tensor, but "
                    f"it changed the shape from {shape_before} to "
                    f"{out.shape}."
                )
        return out

    def na_forward(self, feat: Tensor) -> Tensor:
        r"""Replace NaN values in input :obj:`Tensor` given
        :obj:`na_mode`.

        Args:
            feat: Input :obj:`Tensor`.

        Returns:
            Tensor: Output :obj:`Tensor` with NaNs replaced given
                :obj:`na_mode`.
        """
        if self.na_mode is None:
            return feat

        # Since we are not changing the number of items in each column, it's
        # faster to just clone the values, while reusing the same offset
        # object.
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
            if self.na_mode == NAMode.MOST_FREQUENT:
                fill_value = self.stats_list[col][StatType.MOST_FREQUENT]
                # counter = Counter(feat[:, col][feat[:, col] != -1].tolist())
                # fill_value = max(counter, key=counter.get)
            elif self.na_mode == NAMode.MEAN:
                fill_value = self.stats_list[col][StatType.MEAN]
                # fill_value = torch.mean(
                # feat[:, col][~torch.isnan(feat[:, col])])
            elif self.na_mode == NAMode.ZERO:
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


class CategoricalTransform(ColTypeTransform):
    r"""An simple embedding look-up based Transform for categorical features.
    It applies :class:`torch.nn.Embedding` for each categorical feature and
    concatenates the output embeddings.
    """

    supported_types = {ColType.CATEGORICAL}

    def __init__(
        self,
        out_channels: int | None = None,
        stats_list: list[dict[StatType, Any]] | None = None,
        col_type: ColType | None = ColType.CATEGORICAL,
        post_module: Module | None = None,
        na_mode: NAMode | None = None,
    ) -> None:
        super().__init__(
            out_channels,
            stats_list,
            col_type,
            post_module,
            na_mode
        )

    def post_init(self):
        r"""This is the actual initialization function."""
        num_categories_list = [0]
        for stats in self.stats_list:
            num_categories = stats[StatType.COUNT]
            num_categories_list.append(num_categories)
        # Single embedding module that stores embeddings of all categories
        # across all categorical columns.
        # 0-th category is for NaN.
        self.emb = Embedding(
            sum(num_categories_list) + 1,
            self.out_channels,
            padding_idx=0,
        )
        # [num_cols, ]
        self.register_buffer(
            "offset",
            torch.cumsum(
                torch.tensor(num_categories_list[:-1], dtype=torch.long), dim=0
            ),
        )
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.emb.reset_parameters()

    def encode_forward(
        self,
        feat: Tensor,
        col_names: list[str] | None = None,
    ) -> Tensor:
        # feat: [batch_size, num_cols]
        # Get NaN mask
        na_mask = feat < 0
        # Increment the index by one not to conflict with the padding idx
        # Also add offset for each column to avoid embedding conflict
        feat = feat + self.offset + 1
        # Use 0th index for NaN
        feat[na_mask] = 0
        # [batch_size, num_cols, channels]
        return self.emb(feat)


class LinearTransform(ColTypeTransform):
    r"""A linear function based Transform for numerical features. It applies
    linear layer :obj:`torch.nn.Linear(1, out_channels)` on each raw numerical
    feature and concatenates the output embeddings. Note that the
    implementation does this for all numerical features in a batched manner.
    """

    supported_types = {ColType.NUMERICAL}

    def __init__(
        self,
        out_channels: int | None = None,
        stats_list: list[dict[StatType, Any]] | None = None,
        col_type: ColType | None = ColType.NUMERICAL,
        post_module: Module | None = None,
        na_mode: NAMode | None = None,
    ):
        super().__init__(
            out_channels,
            stats_list,
            col_type,
            post_module,
            na_mode
        )

    def post_init(self):
        r"""This is the actual initialization function."""
        mean = torch.tensor(
            [stats[StatType.MEAN] for stats in self.stats_list]
        )
        self.register_buffer("mean", mean)
        std = torch.tensor(
            [stats[StatType.STD] for stats in self.stats_list]
        ) + 1e-6
        self.register_buffer("std", std)
        num_cols = len(self.stats_list)
        self.weight = Parameter(torch.empty(num_cols, self.out_channels))
        self.bias = Parameter(torch.empty(num_cols, self.out_channels))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        super().reset_parameters()
        torch.nn.init.normal_(self.weight, std=0.01)
        torch.nn.init.zeros_(self.bias)

    def encode_forward(
        self,
        feat: Tensor,
        col_names: list[str] | None = None,
    ) -> Tensor:
        # feat: [batch_size, num_cols]
        feat = (feat - self.mean) / self.std
        # [batch_size, num_cols], [channels, num_cols]
        # -> [batch_size, num_cols, channels]
        x_lin = torch.einsum("ij,jk->ijk", feat, self.weight)
        # [batch_size, num_cols, channels] + [num_cols, channels]
        # -> [batch_size, num_cols, channels]
        x = x_lin + self.bias
        return x


class StackTransform(ColTypeTransform):
    r"""Simply stack input numerical features of shape
    :obj:`[batch_size, num_cols]` into
    :obj:`[batch_size, num_cols, out_channels]`.
    """

    supported_types = {ColType.NUMERICAL}

    def __init__(
        self,
        out_channels: int | None = None,
        stats_list: list[dict[StatType, Any]] | None = None,
        col_type: ColType | None = ColType.NUMERICAL,
        post_module: Module | None = None,
        na_mode: NAMode | None = None,
    ) -> None:
        super().__init__(
            out_channels,
            stats_list,
            col_type,
            post_module,
            na_mode
        )

    def post_init(self) -> None:
        mean = torch.tensor(
            [stats[StatType.MEAN] for stats in self.stats_list])
        self.register_buffer("mean", mean)
        std = (torch.tensor([stats[StatType.STD]
                             for stats in self.stats_list]) + 1e-6)
        self.register_buffer("std", std)

    def reset_parameters(self) -> None:
        super().reset_parameters()

    def encode_forward(
        self,
        feat: Tensor,
        col_names: list[str] | None = None,
    ) -> Tensor:
        # feat: [batch_size, num_cols]
        feat = (feat - self.mean) / self.std
        # x: [batch_size, num_cols, out_channels]
        x = feat.unsqueeze(2).repeat(1, 1, self.out_channels)
        return x


def NumericalTransform(
    type: str = None,
    out_channels: int | None = None,
    stats_list: list[dict[StatType, Any]] | None = None,
    col_type: ColType | None = None,
    post_module: Module | None = None,
    na_mode: NAMode | None = None,
) -> ColTypeTransform:
    r"""A selector that allows users to freely
    choose how to encode numerical features.
    """

    if type == "linear":
        return LinearTransform(
            out_channels=out_channels,
            stats_list=stats_list,
            col_type=col_type,
            na_mode=na_mode,
            post_module=post_module,
        )
    if type == "stack":
        return StackTransform(
            out_channels=out_channels,
            stats_list=stats_list,
            col_type=col_type,
            na_mode=na_mode,
            post_module=post_module,
        )
