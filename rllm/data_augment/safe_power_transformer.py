#  Copyright (c) Prior Labs GmbH 2025.

"""Safe Power Transformer that handles edge cases."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import scipy.optimize
from sklearn.preprocessing import PowerTransformer
from typing_extensions import Self


class SafePowerTransformer(PowerTransformer):
    """Power Transformer which reverts features back to their original values.

    Reverts features if they are transformed to large values or the output
    column does not have unit variance. This happens e.g. when the input
    data has a large number of outliers.
    """

    def __init__(
        self,
        variance_threshold: float = 1e-3,
        large_value_threshold: float = 100,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.variance_threshold = variance_threshold
        self.large_value_threshold = large_value_threshold

        self.revert_indices_ = None

    def _find_features_to_revert_because_of_failure(
        self,
        transformed_X: np.ndarray,
    ) -> None:
        # Calculate the variance for each feature in the transformed data
        variances = np.nanvar(transformed_X, axis=0)

        # Identify features where the variance is not close to 1
        mask = np.abs(variances - 1) > self.variance_threshold
        non_unit_variance_indices = np.where(mask)[0]

        # Identify features with values greater than the large_value_threshold
        large_value_indices = np.any(transformed_X > self.large_value_threshold, axis=0)
        large_value_indices = np.nonzero(large_value_indices)[0]

        # Identify features to revert based on either condition
        self.revert_indices_ = np.unique(
            np.concatenate([non_unit_variance_indices, large_value_indices]),
        )

    def _yeo_johnson_optimize(self, x: np.ndarray) -> float:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r"overflow encountered",
                    category=RuntimeWarning,
                )
                return super()._yeo_johnson_optimize(x)  # type: ignore
        except scipy.optimize._optimize.BracketError:
            return np.nan

    def _yeo_johnson_transform(self, x: np.ndarray, lmbda: float) -> np.ndarray:
        if np.isnan(lmbda):
            return x

        return super()._yeo_johnson_transform(x, lmbda)  # type: ignore

    def _revert_failed_features(
        self,
        transformed_X: np.ndarray,
        original_X: np.ndarray,
    ) -> np.ndarray:
        # Replace these features with the original features
        if self.revert_indices_ and (self.revert_indices_) > 0:
            transformed_X[:, self.revert_indices_] = original_X[:, self.revert_indices_]

        return transformed_X

    def fit(self, X: np.ndarray, y: Any | None = None) -> Self:
        super().fit(X, y)

        # Check and revert features as necessary
        self._find_features_to_revert_because_of_failure(super().transform(X))  # type: ignore
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        transformed_X = super().transform(X)
        return self._revert_failed_features(transformed_X, X)  # type: ignore
