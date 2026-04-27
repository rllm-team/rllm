"""Utilities for fitting ensemble augmentation pipelines."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import overload

import numpy as np

from rllm.data_augment._add_fingerprint_features_augmentor import (
    AddFingerprintFeaturesAugmentor,
)
from rllm.data_augment._add_svd_features_augmentor import AddSVDFeaturesAugmentor
from rllm.data_augment._encode_categorical_features_augmentor import (
    EncodeCategoricalFeaturesAugmentor,
)
from rllm.data_augment._nan_handling_polynomial_features_augmentor import (
    NanHandlingPolynomialFeaturesAugmentor,
)
from rllm.data_augment._remove_constant_features_augmentor import (
    RemoveConstantFeaturesAugmentor,
)
from rllm.data_augment._reshape_feature_distributions_augmentor import (
    ReshapeFeatureDistributionsAugmentor,
)
from rllm.data_augment._shuffle_features_augmentor import ShuffleFeaturesAugmentor
from rllm.data_augment.augmentor_pipeline import AugmentorPipeline
from rllm.data_augment.data_augmentor import DataAugmentor
from rllm.data_augment.ensemble_config import (
    ClassifierEnsembleConfig,
    EnsembleConfig,
    RegressorEnsembleConfig,
)
from rllm.data_augment.utils import infer_random_state


class EnsembleAugmentor:
    """An augmentor composed of multiple ``AugmentorPipeline`` objects."""

    def __init__(
        self,
        configs: Sequence[EnsembleConfig],
        *,
        random_state: int | np.random.Generator | None = None,
    ):
        self.configs = list(configs)
        self.random_state = random_state
        self.augmentor_pipelines = self._build_augmentor_pipelines()

    def __len__(self) -> int:
        return len(self.augmentor_pipelines)

    @overload
    def __getitem__(self, index: int) -> AugmentorPipeline: ...

    @overload
    def __getitem__(self, index: slice) -> list[AugmentorPipeline]: ...

    def __getitem__(
        self,
        index: int | slice,
    ) -> AugmentorPipeline | list[AugmentorPipeline]:
        return self.augmentor_pipelines[index]

    def __iter__(self) -> Iterator[AugmentorPipeline]:
        return iter(self.augmentor_pipelines)

    def _build_augmentor_pipelines(self) -> list[AugmentorPipeline]:
        """Build one augmentation pipeline per ensemble config."""
        _, rng = infer_random_state(self.random_state)
        seeds = rng.integers(0, np.iinfo(np.int32).max, len(self.configs))
        return [
            self.to_pipeline(config, random_state=seed)
            for config, seed in zip(self.configs, seeds)
        ]

    @staticmethod
    def to_pipeline(
        config: EnsembleConfig,
        *,
        random_state: int | np.random.Generator | None = None,
    ) -> AugmentorPipeline:
        """Convert an ensemble configuration to an augmenting pipeline."""
        augmentors: list[DataAugmentor] = []

        if isinstance(config.polynomial_features, int):
            assert config.polynomial_features > 0, "Poly. features to add must be >0!"
            use_poly_features = True
            max_poly_features = config.polynomial_features
        elif config.polynomial_features == "all":
            use_poly_features = True
            max_poly_features = None
        elif config.polynomial_features == "no":
            use_poly_features = False
            max_poly_features = None
        else:
            raise ValueError(
                f"Invalid polynomial_features value: {config.polynomial_features}"
            )

        if use_poly_features:
            augmentors.append(
                NanHandlingPolynomialFeaturesAugmentor(
                    max_features=max_poly_features,
                    random_state=random_state,
                )
            )

        augmentors.extend(
            [
                RemoveConstantFeaturesAugmentor(),
                ReshapeFeatureDistributionsAugmentor(
                    transform_name=config.augmentor,
                    append_to_original=config.append_original,
                    subsample_features=config.subsample_features,
                    transform_sequence=config.transform_sequence,
                    apply_to_categorical=(config.categorical_name == "numeric"),
                    random_state=random_state,
                ),
                EncodeCategoricalFeaturesAugmentor(
                    config.categorical_name,
                    random_state=random_state,
                ),
            ]
        )

        if (
            config.global_transformer_name is not None
            and config.global_transformer_name != "None"
        ):
            augmentors.append(
                AddSVDFeaturesAugmentor(
                    global_transformer_name=config.global_transformer_name,
                    random_state=random_state,
                )
            )

        if config.add_fingerprint_feature:
            augmentors.append(
                AddFingerprintFeaturesAugmentor(random_state=random_state)
            )

        augmentors.append(
            ShuffleFeaturesAugmentor(
                shuffle_method=config.feature_shift_decoder,
                shuffle_index=config.feature_shift_count,
                random_state=random_state,
            )
        )
        return AugmentorPipeline(augmentors)

    def fit_single_augmentation(
        self,
        ensemble_index: int,
        augmentor_pipeline: AugmentorPipeline,
        X_train: np.ndarray,
        y_train: np.ndarray,
        *,
        cat_ix: list[int],
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        list[int],
    ]:
        """Fit augmentation pipeline for a single ensemble configuration."""
        config = self.configs[ensemble_index]
        if config.subsample_ix is not None:
            X_train = X_train[config.subsample_ix].copy()
            y_train = y_train[config.subsample_ix].copy()
        else:
            X_train = X_train.copy()
            y_train = y_train.copy()

        X_train_aug, cat_features = augmentor_pipeline.fit_transform(X_train, cat_ix)

        if isinstance(config, RegressorEnsembleConfig):
            if config.target_transform is not None:
                y_train = config.target_transform.fit_transform(
                    y_train.reshape(-1, 1)
                ).ravel()
        elif isinstance(config, ClassifierEnsembleConfig):
            if config.class_permutation is not None:
                y_train = config.class_permutation[y_train]
        else:
            raise ValueError(f"Invalid ensemble config type: {type(config)}")

        return (X_train_aug, y_train, cat_features)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        *,
        cat_ix: list[int] | None = None,
    ) -> Iterator[
        tuple[
            np.ndarray,
            np.ndarray,
            list[int],
        ]
    ]:
        """Fit augmentation pipelines for all ensemble members."""
        if cat_ix is None:
            cat_ix = []

        for ensemble_index, augmentor_pipeline in enumerate(self.augmentor_pipelines):
            yield self.fit_single_augmentation(
                ensemble_index,
                augmentor_pipeline,
                X_train,
                y_train,
                cat_ix=cat_ix,
            )
