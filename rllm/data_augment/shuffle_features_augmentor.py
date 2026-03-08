#  Copyright (c) Prior Labs GmbH 2025.

"""Shuffle features in the data."""

from __future__ import annotations

from typing import Literal

import numpy as np
from typing_extensions import override

from rllm.data_augment.data_augmentor import DataAugmentor
from rllm.data_augment.utils import infer_random_state


class ShuffleFeaturesAugmentor(DataAugmentor):
    """Shuffle the features in the data."""

    def __init__(
        self,
        shuffle_method: Literal["shuffle", "rotate"] | None = "rotate",
        shuffle_index: int = 0,
        random_state: int | np.random.Generator | None = None,
    ):
        super().__init__()
        self.random_state = random_state
        self.shuffle_method = shuffle_method
        self.shuffle_index = shuffle_index

        self.index_permutation_: list[int] | None = None

    @override
    def _fit(self, X: np.ndarray, categorical_features: list[int]) -> list[int]:
        static_seed, rng = infer_random_state(self.random_state)
        if self.shuffle_method == "rotate":
            index_permutation = np.roll(
                np.arange(X.shape[1]),
                self.shuffle_index,
            ).tolist()
        elif self.shuffle_method == "shuffle":
            index_permutation = rng.permutation(X.shape[1]).tolist()
        elif self.shuffle_method is None:
            index_permutation = np.arange(X.shape[1]).tolist()
        else:
            raise ValueError(f"Unknown shuffle method {self.shuffle_method}")

        self.index_permutation_ = index_permutation

        return [
            new_idx
            for new_idx, idx in enumerate(index_permutation)
            if idx in categorical_features
        ]

    @override
    def _transform(self, X: np.ndarray, *, is_test: bool = False) -> np.ndarray:
        assert self.index_permutation_ is not None, "You must call fit first"
        assert (
            len(self.index_permutation_) == X.shape[1]
        ), "The number of features must not change after fit"
        print(
            f"Shuffling features using method {self.shuffle_method,self.index_permutation_,X[:, self.index_permutation_].shape}"
        )
        return X[:, self.index_permutation_]
