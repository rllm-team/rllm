#  Copyright (c) Prior Labs GmbH 2025.

"""Reshape feature distributions using different transformations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, QuantileTransformer
from typing_extensions import override

from rllm.data_augment.data_augmentor import DataAugmentor
from rllm.data_augment.utils import _identity, infer_random_state

if TYPE_CHECKING:
    from sklearn.base import TransformerMixin


class ReshapeFeatureDistributionsAugmentor(DataAugmentor):
    """Reshape the feature distributions using different transformations."""

    APPEND_TO_ORIGINAL_THRESHOLD = 500

    @staticmethod
    def make_quantile_transformer(
        *,
        num_examples: int,
        n_quantiles: int,
        output_distribution: str,
        random_state: int | None,
        subsample: int = 100_000,
    ) -> QuantileTransformer:
        return QuantileTransformer(
            output_distribution=output_distribution,
            n_quantiles=max(1, min(n_quantiles, num_examples, int(subsample * 0.2))),
            subsample=subsample,
            random_state=random_state,
        )

    @staticmethod
    def get_feature_transformers(
        num_examples: int,
        random_state: int | None = None,
    ) -> dict[str, TransformerMixin | Pipeline]:
        return {
            "quantile_uni_coarse": (
                ReshapeFeatureDistributionsAugmentor.make_quantile_transformer(
                    num_examples=num_examples,
                    output_distribution="uniform",
                    n_quantiles=max(num_examples // 10, 2),
                    random_state=random_state,
                )
            ),
            "quantile_norm_coarse": (
                ReshapeFeatureDistributionsAugmentor.make_quantile_transformer(
                    num_examples=num_examples,
                    output_distribution="normal",
                    n_quantiles=max(num_examples // 10, 2),
                    random_state=random_state,
                )
            ),
            "quantile_uni": (
                ReshapeFeatureDistributionsAugmentor.make_quantile_transformer(
                    num_examples=num_examples,
                    output_distribution="uniform",
                    n_quantiles=max(num_examples // 5, 2),
                    random_state=random_state,
                )
            ),
            "quantile_norm": (
                ReshapeFeatureDistributionsAugmentor.make_quantile_transformer(
                    num_examples=num_examples,
                    output_distribution="normal",
                    n_quantiles=max(num_examples // 5, 2),
                    random_state=random_state,
                )
            ),
            "quantile_uni_fine": (
                ReshapeFeatureDistributionsAugmentor.make_quantile_transformer(
                    num_examples=num_examples,
                    output_distribution="uniform",
                    n_quantiles=num_examples,
                    random_state=random_state,
                )
            ),
            "quantile_norm_fine": (
                ReshapeFeatureDistributionsAugmentor.make_quantile_transformer(
                    num_examples=num_examples,
                    output_distribution="normal",
                    n_quantiles=num_examples,
                    random_state=random_state,
                )
            ),
            "none": FunctionTransformer(_identity),
        }

    def __init__(
        self,
        *,
        transform_name: str = "quantile_uni",
        apply_to_categorical: bool = False,
        append_to_original: bool | str = False,
        subsample_features: float = -1,
        transform_sequence: tuple[str, ...] | None = None,
        random_state: int | np.random.Generator | None = None,
    ):
        super().__init__()
        self.transform_name = transform_name
        self.apply_to_categorical = apply_to_categorical
        self.append_to_original = append_to_original
        self.random_state = random_state
        self.subsample_features = subsample_features
        self.transform_sequence = transform_sequence
        self.transformer_: Pipeline | ColumnTransformer | None = None

    def _set_transformer_and_cat_ix(  # noqa: PLR0912
        self,
        n_samples: int,
        n_features: int,
        categorical_features: list[int],
    ) -> tuple[Pipeline | ColumnTransformer, list[int]]:
        static_seed, rng = infer_random_state(self.random_state)

        feature_transformers = self.get_feature_transformers(
            n_samples,
            random_state=static_seed,
        )
        if self.subsample_features > 0:
            if isinstance(self.subsample_features, int):
                if n_features > self.subsample_features:
                    subsample_features = self.subsample_features
                    self.subsampled_features_ = rng.choice(
                        list(range(n_features)),
                        subsample_features,
                        replace=False,
                    )
                    categorical_features = [
                        new_idx
                        for new_idx, idx in enumerate(self.subsampled_features_)
                        if idx in categorical_features
                    ]
                    n_features = subsample_features
                else:
                    self.subsampled_features_ = np.arange(n_features)
            else:
                subsample_features = int(self.subsample_features * n_features) + 1
                subsample_features = min(subsample_features, n_features)
                self.subsampled_features_ = rng.choice(
                    list(range(n_features)),
                    subsample_features,
                    replace=False,
                )
                categorical_features = [
                    new_idx
                    for new_idx, idx in enumerate(self.subsampled_features_)
                    if idx in categorical_features
                ]
                n_features = subsample_features
        else:
            self.subsampled_features_ = np.arange(n_features)

        all_feats_ix = list(range(n_features))
        transformers = []

        numerical_ix = [i for i in range(n_features) if i not in categorical_features]

        if self.append_to_original == "auto":
            max_features_per_estimator = (
                int(self.subsample_features)
                if isinstance(self.subsample_features, int)
                and self.subsample_features > 0
                else self.APPEND_TO_ORIGINAL_THRESHOLD
            )
            self.append_to_original = bool(
                n_features < self.APPEND_TO_ORIGINAL_THRESHOLD
                and n_features < (max_features_per_estimator / 2)
            )

        # -------- Append to original ------
        # If we append to original, all the categorical indices are kept in place
        # as the first transform is a passthrough on the whole X as it is above
        if self.append_to_original and self.apply_to_categorical:
            trans_ixs = categorical_features + numerical_ix
            transformers.append(("original", "passthrough", all_feats_ix))
            cat_ix = categorical_features  # Exist as they are in original

        elif self.append_to_original and not self.apply_to_categorical:
            trans_ixs = numerical_ix
            # Includes the categoricals passed through
            transformers.append(("original", "passthrough", all_feats_ix))
            cat_ix = categorical_features  # Exist as they are in original

        # -------- Don't append to original ------
        # We only have categorical indices if we don't transform them
        # The first transformer will be a passthrough on the categorical indices
        # Making them the first
        elif not self.append_to_original and self.apply_to_categorical:
            trans_ixs = categorical_features + numerical_ix
            cat_ix = []  # We have none left, they've been transformed

        elif not self.append_to_original and not self.apply_to_categorical:
            trans_ixs = numerical_ix
            transformers.append(("cats", "passthrough", categorical_features))
            cat_ix = list(range(len(categorical_features)))  # They are at start

        else:
            raise ValueError(
                f"Unrecognized combination of {self.apply_to_categorical=}"
                f" and {self.append_to_original=}",
            )

        if self.transform_sequence:
            steps = []
            for idx, name in enumerate(self.transform_sequence):
                if name not in feature_transformers:
                    raise KeyError(f"Unknown transform in sequence: {name}")
                steps.append((f"seq_{idx}_{name}", feature_transformers[name]))
            feature_transformer = Pipeline(steps)
        else:
            if self.transform_name not in feature_transformers:
                raise KeyError(f"Unknown transform: {self.transform_name}")
            feature_transformer = feature_transformers[self.transform_name]
        transformers.append(("feat_transform", feature_transformer, trans_ixs))

        transformer = ColumnTransformer(
            transformers,
            remainder="drop",
            sparse_threshold=0.0,  # No sparse
        )

        self.transformer_ = transformer

        return transformer, cat_ix

    @override
    def _fit(self, X: np.ndarray, categorical_features: list[int]) -> list[int]:
        n_samples, n_features = X.shape
        transformer, cat_ix = self._set_transformer_and_cat_ix(
            n_samples,
            n_features,
            categorical_features,
        )
        transformer.fit(X[:, self.subsampled_features_])
        self.categorical_features_after_transform_ = cat_ix
        self.transformer_ = transformer
        return cat_ix

    @override
    def fit_transform(
        self,
        X: np.ndarray,
        categorical_features: list[int],
    ) -> tuple[np.ndarray, list[int]]:
        n_samples, n_features = X.shape
        transformer, cat_ix = self._set_transformer_and_cat_ix(
            n_samples,
            n_features,
            categorical_features,
        )
        Xt = transformer.fit_transform(X[:, self.subsampled_features_])
        self.categorical_features_after_transform_ = cat_ix
        self.transformer_ = transformer
        return (Xt, cat_ix)  # type: ignore

    @override
    def _transform(self, X: np.ndarray, *, is_test: bool = False) -> np.ndarray:
        assert self.transformer_ is not None, "You must call fit first"
        return self.transformer_.transform(X[:, self.subsampled_features_])  # type: ignore
