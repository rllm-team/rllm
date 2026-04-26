"""Utilities for fitting ensemble augmentation pipelines."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

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
    """Fit a collection of ensemble configurations into augmentation pipelines."""

    def __init__(self, configs: Sequence[EnsembleConfig]):
        self.configs = list(configs)

    @staticmethod
    def to_pipeline(
        config: EnsembleConfig,
        *,
        random_state: int | np.random.Generator | None,
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
                    transform_name=config.augment_config.name,
                    append_to_original=config.augment_config.append_original,
                    subsample_features=config.augment_config.subsample_features,
                    transform_sequence=config.augment_config.transform_sequence,
                    apply_to_categorical=(
                        config.augment_config.categorical_name == "numeric"
                    ),
                    random_state=random_state,
                ),
                EncodeCategoricalFeaturesAugmentor(
                    config.augment_config.categorical_name,
                    random_state=random_state,
                ),
            ]
        )

        if (
            config.augment_config.global_transformer_name is not None
            and config.augment_config.global_transformer_name != "None"
        ):
            augmentors.append(
                AddSVDFeaturesAugmentor(
                    global_transformer_name=(
                        config.augment_config.global_transformer_name
                    ),
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
        config: EnsembleConfig,
        X_train: np.ndarray,
        y_train: np.ndarray,
        random_state: int | np.random.Generator | None = None,
        *,
        cat_ix: list[int],
    ) -> tuple[
        EnsembleConfig,
        AugmentorPipeline,
        np.ndarray,
        np.ndarray,
        list[int],
    ]:
        """Fit augmentation pipeline for a single ensemble configuration."""
        static_seed, _ = infer_random_state(random_state)
        if config.subsample_ix is not None:
            X_train = X_train[config.subsample_ix].copy()
            y_train = y_train[config.subsample_ix].copy()
        else:
            X_train = X_train.copy()
            y_train = y_train.copy()

        augmentor = self.to_pipeline(config, random_state=static_seed)
        X_train_aug, cat_features = augmentor.fit_transform(X_train, cat_ix)

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

        return (config, augmentor, X_train_aug, y_train, cat_features)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        *,
        random_state: int | np.random.Generator | None,
        cat_ix: list[int] | None = None,
    ) -> Iterator[
        tuple[
            EnsembleConfig,
            AugmentorPipeline,
            np.ndarray,
            np.ndarray,
            list[int],
        ]
    ]:
        """Fit augmentation pipelines for all ensemble members."""
        if cat_ix is None:
            cat_ix = []

        _, rng = infer_random_state(random_state)
        seeds = rng.integers(0, np.iinfo(np.int32).max, len(self.configs))
        for config, seed in zip(self.configs, seeds):
            yield self.fit_single_augmentation(
                config,
                X_train,
                y_train,
                seed,
                cat_ix=cat_ix,
            )
