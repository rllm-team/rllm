#  Copyright (c) Prior Labs GmbH 2025.

"""Base class for data augmentation components."""

from __future__ import annotations

from abc import abstractmethod
from typing import NamedTuple

import numpy as np
from typing_extensions import Self, override


class _TransformResult(NamedTuple):
    """Result of a transformation containing the transformed data and categorical features."""

    X: np.ndarray
    categorical_features: list[int]


# TODO(eddiebergman): I'm sure there's a way to handle this when using dataframes.
class DataAugmentor:
    """Base class for data augmentation components.

    It's main abstraction is really just to provide categorical indices along the
    pipeline.
    """

    categorical_features_after_transform_: list[int]

    def fit_transform(
        self,
        X: np.ndarray,
        categorical_features: list[int],
    ) -> _TransformResult:
        self.fit(X, categorical_features)
        # TODO(eddiebergman): If we could get rid of this... anywho, needed for
        # the AddFingerPrint
        result = self._transform(X, is_test=False)
        return _TransformResult(result, self.categorical_features_after_transform_)

    @abstractmethod
    def _fit(self, X: np.ndarray, categorical_features: list[int]) -> list[int]:
        """Underlying method of the augmentor to implement by subclassses.

        Args:
            X: 2d array of shape (n_samples, n_features)
            categorical_features: list of indices of categorical feature.

        Returns:
            list of indices of categorical features after the transform.
        """
        raise NotImplementedError

    def fit(self, X: np.ndarray, categorical_features: list[int]) -> Self:
        """Fits the augmentor.

        Args:
            X: 2d array of shape (n_samples, n_features)
            categorical_features: list of indices of categorical feature.
        """
        self.categorical_features_after_transform_ = self._fit(X, categorical_features)
        assert self.categorical_features_after_transform_ is not None, (
            "_fit should have returned a list of the indexes of the categorical"
            "features after the transform."
        )
        return self

    @abstractmethod
    def _transform(self, X: np.ndarray, *, is_test: bool = False) -> np.ndarray:
        """Underlying method of the augmentor to implement by subclassses.

        Args:
            X: 2d array of shape (n_samples, n_features)
            is_test: Should be removed, used for the `AddFingerPrint` augmentor.

        Returns:
            2d np.ndarray of shape (n_samples, new n_features)
        """
        raise NotImplementedError

    def transform(self, X: np.ndarray) -> _TransformResult:
        """Transforms the data.

        Args:
            X: 2d array of shape (n_samples, n_features).
        """
        # TODO: Get rid of this, it's always test in `transform`
        result = self._transform(X, is_test=True)
        return _TransformResult(result, self.categorical_features_after_transform_)
