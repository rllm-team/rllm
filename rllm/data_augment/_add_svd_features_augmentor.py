#  Copyright (c) Prior Labs GmbH 2025.

"""Add SVD features as a standalone augmentation step."""

from __future__ import annotations

from typing import Literal

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from typing_extensions import override

from rllm.data_augment.data_augmentor import DataAugmentor
from rllm.data_augment.utils import infer_random_state, make_standard_scaler_safe


def get_svd_features_transformer(
    global_transformer_name: Literal["svd", "svd_quarter_components"],
    n_samples: int,
    n_features: int,
    random_state: int | None = None,
) -> Pipeline:
    if global_transformer_name == "svd":
        divisor = 2
    elif global_transformer_name == "svd_quarter_components":
        divisor = 4
    else:
        raise ValueError(f"Invalid global transformer name: {global_transformer_name}.")

    n_components = max(1, min(n_samples // 10 + 1, n_features // divisor))
    return Pipeline(
        steps=[
            (
                "save_standard",
                make_standard_scaler_safe(("standard", StandardScaler(with_mean=False))),
            ),
            (
                "svd",
                TruncatedSVD(
                    algorithm="arpack",
                    n_components=n_components,
                    random_state=random_state,
                ),
            ),
        ],
    )


class AddSVDFeaturesAugmentor(DataAugmentor):
    """Append low-rank SVD projection features to the existing feature matrix."""

    def __init__(
        self,
        global_transformer_name: Literal["svd", "svd_quarter_components"] = "svd",
        random_state: int | np.random.Generator | None = None,
    ):
        super().__init__()
        self.global_transformer_name = global_transformer_name
        self.random_state = random_state
        self.transformer_: Pipeline | None = None
        self.is_no_op_: bool = False

    @override
    def _fit(self, X: np.ndarray, categorical_features: list[int]) -> list[int]:
        n_samples, n_features = X.shape
        if n_features < 2:
            self.is_no_op_ = True
            self.transformer_ = None
            return categorical_features

        static_seed, _ = infer_random_state(self.random_state)
        self.transformer_ = get_svd_features_transformer(
            self.global_transformer_name,
            n_samples,
            n_features,
            random_state=static_seed,
        )
        self.transformer_.fit(X)
        self.is_no_op_ = False
        return categorical_features

    @override
    def _transform(self, X: np.ndarray, *, is_test: bool = False) -> np.ndarray:
        if self.is_no_op_ or self.transformer_ is None:
            return X
        X_added = self.transformer_.transform(X)
        return np.concatenate([X, X_added], axis=1)
