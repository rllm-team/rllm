#  Copyright (c) Prior Labs GmbH 2025.

"""Remove constant features from the data."""

from __future__ import annotations

import numpy as np
from typing_extensions import override

from rllm.data_augment.data_augmentor import DataAugmentor


class RemoveConstantFeaturesAugmentor(DataAugmentor):
    """Remove features that are constant in the training data."""

    def __init__(self) -> None:
        super().__init__()
        self.sel_: list[bool] | None = None

    @override
    def _fit(self, X: np.ndarray, categorical_features: list[int]) -> list[int]:
        sel_ = ((X[0:1, :] == X).mean(axis=0) < 1.0).tolist()

        if not any(sel_):
            raise ValueError(
                "All features are constant and would have been removed!"
                " Unable to predict using TabPFN.",
            )
        self.sel_ = sel_

        return [
            new_idx
            for new_idx, idx in enumerate(np.where(sel_)[0])
            if idx in categorical_features
        ]

    @override
    def _transform(self, X: np.ndarray, *, is_test: bool = False) -> np.ndarray:
        assert self.sel_ is not None, "You must call fit first"
        return X[:, self.sel_]
