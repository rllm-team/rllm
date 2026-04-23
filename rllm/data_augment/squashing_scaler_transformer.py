"""Implementation of the default TabPFN squashing scaler."""

from __future__ import annotations

import numbers
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.preprocessing import RobustScaler
from sklearn.utils.validation import FLOAT_DTYPES, check_is_fitted

try:
    from sklearn.utils.validation import validate_data as sklearn_validate_data
except ImportError:
    sklearn_validate_data = None


def _validate_data(estimator: BaseEstimator, **kwargs: Any) -> Any:
    if sklearn_validate_data is not None:
        return sklearn_validate_data(estimator, **kwargs)

    if "ensure_all_finite" in kwargs:
        force_all_finite = kwargs.pop("ensure_all_finite")
    else:
        force_all_finite = True
    return estimator._validate_data(**kwargs, force_all_finite=force_all_finite)


def _mask_inf(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if (mask_inf := np.isinf(X)).any():
        sign = np.sign(X)
        X = np.where(mask_inf, np.nan, X)
        mask_inf = mask_inf.astype(X.dtype) * sign

    return X, mask_inf


def _set_zeros(X: np.ndarray, zero_cols: np.ndarray) -> np.ndarray:
    mask = np.isfinite(X)
    mask[:, ~zero_cols] = False
    X[mask] = 0.0
    return X


def _soft_clip(
    X: np.ndarray,
    max_absolute_value: float,
    mask_inf: np.ndarray,
) -> np.ndarray:
    X = X / np.sqrt(1 + (X / max_absolute_value) ** 2)
    X = np.where(mask_inf == 1, max_absolute_value, X)
    return np.where(mask_inf == -1, -max_absolute_value, X)


class _MinMaxScaler(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    def fit(self, X: np.ndarray, y: None | np.ndarray = None) -> "_MinMaxScaler":
        del y
        eps = np.finfo("float32").tiny
        self.median_ = np.nanmedian(X, axis=0)
        self.scale_ = 2 / (np.nanmax(X, axis=0) - np.nanmin(X, axis=0) + eps)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ["median_", "scale_"])
        return self.scale_ * (X - self.median_)


class SquashingScaler(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    robust_scaler_: RobustScaler | None
    minmax_scaler_: _MinMaxScaler | None
    robust_cols_: np.ndarray
    minmax_cols_: np.ndarray
    zero_cols_: np.ndarray

    def __init__(
        self,
        max_absolute_value: float = 3.0,
        quantile_range: tuple[float, float] = (25.0, 75.0),
    ) -> None:
        super().__init__()
        self.max_absolute_value = max_absolute_value
        self.quantile_range = quantile_range

    def fit(self, X: np.ndarray, y: None | np.ndarray = None) -> "SquashingScaler":
        del y

        if not (
            isinstance(self.max_absolute_value, numbers.Number)
            and np.isfinite(self.max_absolute_value)
            and self.max_absolute_value > 0
        ):
            raise ValueError(
                f"Got max_absolute_value={self.max_absolute_value!r}, but expected a "
                "positive finite number."
            )

        X = _validate_data(
            self,
            X=X,  # type: ignore[arg-type]
            reset=True,
            dtype=FLOAT_DTYPES,
            accept_sparse=False,
            ensure_2d=True,
            ensure_all_finite=False,
        )
        X, _ = _mask_inf(X)

        zero_cols = np.nanmax(X, axis=0) == np.nanmin(X, axis=0)
        quantiles = np.nanpercentile(X, self.quantile_range, axis=0)
        minmax_cols = quantiles[0, :] == quantiles[1, :]
        minmax_cols = minmax_cols & ~zero_cols
        robust_cols = ~(minmax_cols | zero_cols)

        if robust_cols.any():
            self.robust_scaler_ = RobustScaler(
                with_centering=True,
                with_scaling=True,
                quantile_range=self.quantile_range,
                copy=True,
            )
            self.robust_scaler_ = self.robust_scaler_.fit(X[:, robust_cols])
        else:
            self.robust_scaler_ = None
        self.robust_cols_ = robust_cols

        if minmax_cols.any():
            self.minmax_scaler_ = _MinMaxScaler()
            self.minmax_scaler_ = self.minmax_scaler_.fit(X[:, minmax_cols])
        else:
            self.minmax_scaler_ = None
        self.minmax_cols_ = minmax_cols
        self.zero_cols_ = zero_cols
        return self

    def fit_transform(
        self,
        X: np.ndarray,
        y: None | np.ndarray = None,
        **fit_params: Any,
    ) -> np.ndarray:
        del y
        del fit_params
        self.fit(X)
        return self.transform(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(
            self,
            [
                "robust_scaler_",
                "minmax_scaler_",
                "zero_cols_",
                "robust_cols_",
                "minmax_cols_",
            ],
        )

        X = _validate_data(
            self,
            X=X,  # type: ignore[arg-type]
            reset=False,
            dtype=FLOAT_DTYPES,
            accept_sparse=False,
            ensure_2d=True,
            ensure_all_finite=False,
        )
        X, mask_inf = _mask_inf(X)

        X_tr = X.copy()
        if self.robust_cols_.any():
            assert self.robust_scaler_ is not None
            X_tr[:, self.robust_cols_] = self.robust_scaler_.transform(
                X[:, self.robust_cols_]
            )
        if self.minmax_cols_.any():
            assert self.minmax_scaler_ is not None
            X_tr[:, self.minmax_cols_] = self.minmax_scaler_.transform(
                X[:, self.minmax_cols_]
            )
        if self.zero_cols_.any():
            X_tr = _set_zeros(X_tr, self.zero_cols_)

        return _soft_clip(X_tr, self.max_absolute_value, mask_inf)

