#  Copyright (c) Prior Labs GmbH 2025.

"""Base class for data augmentation components."""

from __future__ import annotations

from abc import abstractmethod

import numpy as np


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
    ) -> tuple[np.ndarray, list[int]]:
        self.fit(X, categorical_features)
        # TODO(eddiebergman): If we could get rid of this... anywho, needed for
        # the AddFingerPrint
        result = self._transform(X, is_test=False)
        return (result, self.categorical_features_after_transform_)

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

    def fit(self, X: np.ndarray, categorical_features: list[int]) -> None:
        """Fits the augmentor.

        Args:
            X: 2d array of shape (n_samples, n_features)
            categorical_features: list of indices of categorical feature.
        """
        self.categorical_features_after_transform_ = self._fit(X, categorical_features)

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

    def transform(self, X: np.ndarray) -> tuple[np.ndarray, list[int]]:
        """Transforms the data.

        Args:
            X: 2d array of shape (n_samples, n_features).
        """
        # TODO: Get rid of this, it's always test in `transform`
        result = self._transform(X, is_test=True)
        return (result, self.categorical_features_after_transform_)
