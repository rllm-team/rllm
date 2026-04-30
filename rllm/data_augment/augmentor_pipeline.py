#  Copyright (c) Prior Labs GmbH 2025.

"""Pipeline that applies data augmentation components in sequence."""

from __future__ import annotations

from collections import UserList
from collections.abc import Sequence

import numpy as np

from rllm.data_augment.data_augmentor import DataAugmentor


class AugmentorPipeline(UserList):
    """A pipeline that applies a sequence of data augmentation components.

    This is very related to sklearn's Pipeline, but it is designed to work with
    categorical_features lists that are always passed on.

    Currently this class is only used once, thus this could also be made
    less general if needed.
    """

    def __init__(self, augmentors: Sequence[DataAugmentor]):
        super().__init__(augmentors)
        self.augmentors = augmentors
        self.categorical_features_: list[int] | None = None

    def fit_transform(
        self,
        X: np.ndarray,
        categorical_features: list[int],
    ) -> tuple[np.ndarray, list[int]]:
        """Fit and transform the data using the fitted pipeline.

        Args:
            X: 2d array of shape (n_samples, n_features)
            categorical_features: list of indices of categorical features.
        """
        input_feature_count = X.shape[1]
        original_categorical_features = list(categorical_features)
        for augmentor in self.augmentors:
            X, categorical_features = augmentor.fit_transform(X, categorical_features)

        if (
            len(categorical_features) == 0
            and len(original_categorical_features) > 0
            and X.shape[1] == input_feature_count
        ):
            categorical_features = original_categorical_features

        self.categorical_features_ = categorical_features
        return (X, categorical_features)

    def transform(self, X: np.ndarray) -> tuple[np.ndarray, list[int]]:
        """Transform the data using the fitted pipeline.

        Args:
            X: 2d array of shape (n_samples, n_features).
        """
        categorical_features = []
        for augmentor in self:
            X, categorical_features = augmentor.transform(X)

        return (X, categorical_features)
