#  Copyright (c) Prior Labs GmbH 2025.

"""Utility functions and constants for feature preprocessing."""

from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from itertools import chain, repeat
from typing import TYPE_CHECKING, TypeVar

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

if TYPE_CHECKING:
    from sklearn.base import TransformerMixin


def infer_random_state(
    random_state: int | np.random.RandomState | np.random.Generator | None,
) -> tuple[int, np.random.Generator]:
    """Infer a stable seed and NumPy Generator from common random-state inputs."""
    if isinstance(random_state, (int, np.integer)):
        np_rng = np.random.default_rng(random_state)
        static_seed = int(random_state)
    elif isinstance(random_state, np.random.RandomState):
        static_seed = int(random_state.randint(0, 2**31))
        np_rng = np.random.default_rng(static_seed)
    elif isinstance(random_state, np.random.Generator):
        np_rng = random_state
        static_seed = int(np_rng.integers(0, 2**31))
    elif random_state is None:
        np_rng = np.random.default_rng()
        static_seed = int(np_rng.integers(0, 2**31))
    else:
        raise ValueError(f"Invalid random_state {random_state}")

    return static_seed, np_rng


def balance(x: Iterable[T], n: int) -> list[T]:
    """Replicate each element from an iterable exactly ``n`` times."""
    return list(chain.from_iterable(repeat(elem, n) for elem in x))


def skew(x: np.ndarray) -> float:
    """Calculate skewness of an array.

    Uses the formula: 3 * (mean - median) / std.
    This is a simplified measure of distribution asymmetry.

    Args:
        x: Input array of numeric values.

    Returns:
        Skewness value as a float.
    """
    return float(3 * (np.nanmean(x, 0) - np.nanmedian(x, 0)) / np.std(x, 0))


def _inf_to_nan_func(x: np.ndarray) -> np.ndarray:
    """Convert infinite values to NaN while preserving existing NaN.

    Args:
        x: Input array potentially containing inf/-inf values.

    Returns:
        Array with all inf and -inf replaced by NaN.
    """
    return np.nan_to_num(x, nan=np.nan, neginf=np.nan, posinf=np.nan)


def _exp_minus_1(x: np.ndarray) -> np.ndarray:
    """Compute exp(x) - 1 element-wise.

    Inverse transformation for log(x + 1).

    Args:
        x: Input array.

    Returns:
        Array with exp(x) - 1 applied element-wise.
    """
    return np.exp(x) - 1  # type: ignore


T = TypeVar("T")


def _identity(x: T) -> T:
    """Identity function that returns input unchanged.

    Used as a no-op transformer or inverse function.

    Args:
        x: Input of any type.

    Returns:
        The input unchanged.
    """
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
    """Wrap a scaler with pre/post-processing to ensure finite values.

    Adds transformers before and after the scaler that convert inf to NaN
    and impute NaN values with mean. This prevents edge cases like division
    by zero or non-finite inputs from breaking the scaling process.

    Args:
        _name_scaler_tuple: Tuple of (name, transformer) to wrap.
        no_name: If True, wraps the tuple in a placeholder step.

    Returns:
        Pipeline with finite-value enforcement around the scaler.
    """
    return Pipeline(
        steps=[
            *[(n + "_pre ", deepcopy(t)) for n, t in _make_finite_transformer],
            ("placeholder", _name_scaler_tuple) if no_name else _name_scaler_tuple,
            *[(n + "_post", deepcopy(t)) for n, t in _make_finite_transformer],
        ],
    )


def make_box_cox_safe(input_transformer: TransformerMixin | Pipeline) -> Pipeline:
    """Wrap Box-Cox transformer to ensure strictly positive input data.

    The Box-Cox transformation requires strictly positive values. This wrapper
    applies MinMaxScaler first to shift data to [0.1, 1] range, ensuring
    positivity. Clipping handles out-of-range test data.

    Args:
        input_transformer: Box-Cox or similar transformer requiring positive data.

    Returns:
        Pipeline with MinMaxScaler followed by the input transformer.
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
    """Add safe StandardScaler after PowerTransformer to handle inf values.

    PowerTransformer can sometimes produce inf values in edge cases, causing
    subsequent StandardScaler to fail. This wrapper adds a safe StandardScaler
    that handles non-finite values gracefully.

    Args:
        input_transformer: Typically a PowerTransformer instance.

    Returns:
        Pipeline with the transformer followed by a safe StandardScaler.
    """
    return Pipeline(
        steps=[
            ("input_transformer", input_transformer),
            ("standard", make_standard_scaler_safe(("standard", StandardScaler()))),
        ],
    )


_CONSTANT = 10**12


def float_hash_arr(arr: np.ndarray) -> float:
    """Hash a numpy array to a normalized float value in [0, 1).

    Converts array to bytes, computes hash, and normalizes to [0, 1) range
    using modulo operation. Used for creating fingerprint features.

    Args:
        arr: Input numpy array to hash.

    Returns:
        Float in range [0, 1) representing the hash.
    """
    b = arr.tobytes()
    _hash = hash(b)
    return _hash % _CONSTANT / _CONSTANT
