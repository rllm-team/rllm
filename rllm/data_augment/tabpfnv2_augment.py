"""Data augmentation utilities for TabPFN v2 classification."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from rllm.data_augment.ensemble_augmentors import (
    EnsembleConfig,
    default_classifier_augmentor_configs,
    fit_augmentation,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from rllm.data_augment.ensemble_augmentors import (
        EnsembleConfig,
        AugmentorConfig,
    )


def prepare_classification_ensemble(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    n_estimators: int = 4,
    subsample_size: int | None = None,
    add_fingerprint_feature: bool = True,
    feature_shift_decoder: str = "shuffle",
    polynomial_features: str = "no",
    class_shift_method: str = "shuffle",
    augmentor_configs: list[AugmentorConfig] | None = None,
    random_state: int = 0,
    n_workers: int = 1,
    parallel_mode: str = "block",
    cat_ix: list[int] | None = None,
) -> Iterator:
    """Prepare classification ensemble configurations and their preprocessing.

    This function generates ensemble configurations for classification tasks
    and fits their preprocessing pipelines.

    Args:
        X_train: Training feature array of shape (n_samples, n_features).
        y_train: Training target array of shape (n_samples,).
        n_estimators: Number of ensemble estimators. Default is 4.
        subsample_size: Size of subsamples for each estimator.
            If None, uses min(10_000, len(X_train)). Default is None.
        add_fingerprint_feature: Whether to add fingerprint features. Default is True.
        feature_shift_decoder: Feature shift method ("shuffle" or other). Default is "shuffle".
        polynomial_features: Whether to use polynomial features ("no" or other). Default is "no".
        class_shift_method: Class shift method ("shuffle" or other). Default is "shuffle".
        augmentor_configs: Custom augmentor configs.
            If None, uses default_classifier_augmentor_configs(). Default is None.
        random_state: Random seed for reproducibility. Default is 0.
        n_workers: Number of workers for parallel preprocessing. Default is 1.
        parallel_mode: Parallelization mode ("block" or other). Default is "block".
        cat_ix: List of categorical feature indices. Default is None.

    Returns:
        Iterator over preprocessed ensemble configurations.
    """
    if subsample_size is None:
        subsample_size = min(10_000, len(X_train))

    if augmentor_configs is None:
        augmentor_configs = default_classifier_augmentor_configs()

    if cat_ix is None:
        cat_ix = []

    # Generate ensemble configurations
    ensemble_configs = EnsembleConfig.generate_for_classification(
        n=n_estimators,
        subsample_size=subsample_size,
        add_fingerprint_feature=add_fingerprint_feature,
        feature_shift_decoder=feature_shift_decoder,
        polynomial_features=polynomial_features,
        max_index=len(X_train),
        augmentor_configs=augmentor_configs,
        class_shift_method=class_shift_method,
        n_classes=int(y_train.max()) + 1,
        random_state=random_state,
    )

    augmentors = fit_augmentation(
        configs=ensemble_configs,
        X_train=X_train,
        y_train=y_train,
        random_state=random_state,
        cat_ix=cat_ix,
    )

    return augmentors
