#  Copyright (c) Prior Labs GmbH 2025.

"""Pipeline that applies data augmentation components in sequence."""

from __future__ import annotations

from collections import UserList
from collections.abc import Sequence

import numpy as np
from typing_extensions import Self

from rllm.data_augment.data_augmentor import (
    DataAugmentor,
    _TransformResult,
)


class AugmentorPipeline(UserList):
    """A pipeline that applies a sequence of data augmentation components.

    This is very related to sklearn's Pipeline, but it is designed to work with
    categorical_features lists that are always passed on.

    Currently this class is only used once, thus this could also be made
    less general if needed.
    """

    def __init__(self, steps: Sequence[DataAugmentor]):
        super().__init__(steps)
        self.steps = steps
        self.categorical_features_: list[int] | None = None

    def fit_transform(
        self,
        X: np.ndarray,
        categorical_features: list[int],
    ) -> _TransformResult:
        """Fit and transform the data using the fitted pipeline.

        Args:
            X: 2d array of shape (n_samples, n_features)
            categorical_features: list of indices of categorical features.
        """
        for step in self.steps:
            X, categorical_features = step.fit_transform(X, categorical_features)
            assert isinstance(categorical_features, list), (
                f"The {step=} must return list of categorical features,"
                f" but {type(step)} returned {categorical_features}"
            )

        self.categorical_features_ = categorical_features
        return _TransformResult(X, categorical_features)

    def fit(self, X: np.ndarray, categorical_features: list[int]) -> Self:
        """Fit all the augmentation components in the pipeline.

        Args:
            X: 2d array of shape (n_samples, n_features)
            categorical_features: list of indices of categorical feature.
        """
        assert (
            len(self) > 0
        ), "The AugmentorPipeline must have at least one augmentation component."
        self.fit_transform(X, categorical_features)
        return self

    def transform(self, X: np.ndarray) -> _TransformResult:
        """Transform the data using the fitted pipeline.

        Args:
            X: 2d array of shape (n_samples, n_features).
        """
        assert (
            len(self) > 0
        ), "The AugmentorPipeline must have at least one augmentation component."
        assert self.categorical_features_ is not None, (
            "The AugmentorPipeline must be fit before it" " can be used to transform."
        )
        categorical_features = []
        for step in self:
            X, categorical_features = step.transform(X)

        assert categorical_features == self.categorical_features_, (
            f"Expected categorical features {self.categorical_features_},"
            f"but got {categorical_features}"
        )
        return _TransformResult(X, categorical_features)
