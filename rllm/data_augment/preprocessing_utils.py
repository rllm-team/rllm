#  Copyright (c) Prior Labs GmbH 2025.

"""Utility functions and constants for feature preprocessing."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, TypeVar

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

if TYPE_CHECKING:
    from sklearn.base import TransformerMixin


def skew(x: np.ndarray) -> float:
    """skewness: 3 * (mean - median) / std."""
    return float(3 * (np.nanmean(x, 0) - np.nanmedian(x, 0)) / np.std(x, 0))


def _inf_to_nan_func(x: np.ndarray) -> np.ndarray:
    return np.nan_to_num(x, nan=np.nan, neginf=np.nan, posinf=np.nan)


def _exp_minus_1(x: np.ndarray) -> np.ndarray:
    return np.exp(x) - 1  # type: ignore


T = TypeVar("T")


def _identity(x: T) -> T:
    return x


# Transformer instances
inf_to_nan_transformer = FunctionTransformer(
    func=_inf_to_nan_func,
    inverse_func=_identity,
    check_inverse=False,
)

nan_impute_transformer = SimpleImputer(
    missing_values=np.nan,
    strategy="mean",
    # keep empty features for inverse to function
    keep_empty_features=True,
)
nan_impute_transformer.inverse_transform = (
    _identity  # do not inverse np.nan values.  # type: ignore
)

_make_finite_transformer = [
    ("inf_to_nan", inf_to_nan_transformer),
    ("nan_impute", nan_impute_transformer),
]


def make_standard_scaler_safe(
    _name_scaler_tuple: tuple[str, TransformerMixin],
    *,
    no_name: bool = False,
) -> Pipeline:
    """Make sure that all data that enters and leaves a scaler is finite.

    This is needed in edge cases where, for example, a division by zero
    occurs while scaling or when the input contains not number values.
    """
    return Pipeline(
        steps=[
            *[(n + "_pre ", deepcopy(t)) for n, t in _make_finite_transformer],
            ("placeholder", _name_scaler_tuple) if no_name else _name_scaler_tuple,
            *[(n + "_post", deepcopy(t)) for n, t in _make_finite_transformer],
        ],
    )


def make_box_cox_safe(input_transformer: TransformerMixin | Pipeline) -> Pipeline:
    """Make box cox safe.

    The Box-Cox transformation can only be applied to strictly positive data.
    With first MinMax scaling, we achieve this without loss of function.
    Additionally, for test data, we also need clipping.
    """
    from sklearn.preprocessing import MinMaxScaler

    return Pipeline(
        steps=[
            ("mm", MinMaxScaler(feature_range=(0.1, 1), clip=True)),
            ("box_cox", input_transformer),
        ],
    )


def add_safe_standard_to_safe_power_without_standard(
    input_transformer: TransformerMixin,
) -> Pipeline:
    """In edge cases PowerTransformer can create inf values and similar.

    Then, the post standard scale crashes. This fixes this issue.
    """
    return Pipeline(
        steps=[
            ("input_transformer", input_transformer),
            ("standard", make_standard_scaler_safe(("standard", StandardScaler()))),
        ],
    )


_CONSTANT = 10**12


def float_hash_arr(arr: np.ndarray) -> float:
    """Hash a numpy array to a float value."""
    b = arr.tobytes()
    _hash = hash(b)
    return _hash % _CONSTANT / _CONSTANT
