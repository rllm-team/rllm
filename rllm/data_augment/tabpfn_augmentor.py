"""Data augmentation utilities for the retained TabPFN runtime."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Literal

import numpy as np

from rllm.data_augment.augmentor_pipeline import AugmentorPipeline
from rllm.data_augment.ensemble_augmentor import EnsembleAugmentor
from rllm.data_augment.ensemble_config import (
    AugmentorConfig,
    EnsembleConfig,
)

if TYPE_CHECKING:
    from collections.abc import Iterator


def _build_recipe_configs(
    *,
    append_original_for_categorical: bool | Literal["auto"],
) -> list[AugmentorConfig]:
    return [
        AugmentorConfig(
            "quantile_uni",
            append_original=False,
            categorical_name="numeric",
            global_transformer_name=None,
            subsample_features=680,
            transform_sequence=None,
        ),
        AugmentorConfig(
            "quantile_uni",
            append_original=append_original_for_categorical,
            categorical_name="ordinal_very_common_categories_shuffled",
            global_transformer_name="svd_quarter_components",
            subsample_features=500,
            transform_sequence=None,
        ),
    ]


class TabPFNEnsembleAugmentor(EnsembleAugmentor):
    """Generate and fit TabPFN ensemble augmentation pipelines."""

    def __init__(
        self,
        augmentor_configs: list[AugmentorConfig] | None = None,
        *,
        n_estimators: int = 4,
        task: Literal["classification", "regression"] = "classification",
        subsample_size: int | float | None = None,
        add_fingerprint_feature: bool = True,
        feature_shift_decoder: Literal["shuffle", "rotate"] | None = "shuffle",
        polynomial_features: Literal["no", "all"] | int = "no",
        class_shift_method: Literal["shuffle", "rotate"] | None = "shuffle",
        target_transforms: list | None = None,
    ):
        self.task = task
        if augmentor_configs is None:
            augmentor_configs = [
                AugmentorConfig(
                    "quantile_uni",
                    append_original=False,
                    categorical_name="numeric",
                    global_transformer_name=None,
                    subsample_features=680,
                    transform_sequence=None,
                ),
                AugmentorConfig(
                    "quantile_uni",
                    append_original=False if task == "classification" else "auto",
                    categorical_name="ordinal_very_common_categories_shuffled",
                    global_transformer_name="svd_quarter_components",
                    subsample_features=500,
                    transform_sequence=None,
                ),
            ]

        super().__init__([])
        self.task = task
        self.n_estimators = n_estimators
        self.subsample_size = subsample_size
        self.add_fingerprint_feature = add_fingerprint_feature
        self.feature_shift_decoder = feature_shift_decoder
        self.polynomial_features = polynomial_features
        self.class_shift_method = class_shift_method
        self.augmentor_configs = augmentor_configs
        self.target_transforms = target_transforms

    @classmethod
    def for_classification(
        cls,
        *,
        n_estimators: int = 4,
        subsample_size: int | float | None = None,
        add_fingerprint_feature: bool = True,
        feature_shift_decoder: Literal["shuffle", "rotate"] | None = "shuffle",
        polynomial_features: Literal["no", "all"] | int = "no",
        class_shift_method: Literal["shuffle", "rotate"] | None = "shuffle",
        augmentor_configs: list[AugmentorConfig] | None = None,
    ) -> TabPFNEnsembleAugmentor:
        """Create a TabPFN augmentor for classification."""
        return cls(
            augmentor_configs,
            n_estimators=n_estimators,
            task="classification",
            subsample_size=subsample_size,
            add_fingerprint_feature=add_fingerprint_feature,
            feature_shift_decoder=feature_shift_decoder,
            polynomial_features=polynomial_features,
            class_shift_method=class_shift_method,
        )

    @classmethod
    def for_regression(
        cls,
        *,
        n_estimators: int = 4,
        subsample_size: int | float | None = None,
        add_fingerprint_feature: bool = True,
        feature_shift_decoder: Literal["shuffle", "rotate"] | None = "shuffle",
        polynomial_features: Literal["no", "all"] | int = "no",
        augmentor_configs: list[AugmentorConfig] | None = None,
        target_transforms: list | None = None,
    ) -> TabPFNEnsembleAugmentor:
        """Create a TabPFN augmentor for regression."""
        return cls(
            augmentor_configs,
            n_estimators=n_estimators,
            task="regression",
            subsample_size=subsample_size,
            add_fingerprint_feature=add_fingerprint_feature,
            feature_shift_decoder=feature_shift_decoder,
            polynomial_features=polynomial_features,
            target_transforms=target_transforms,
        )

    def _generate_configs(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        *,
        random_state: int | np.random.Generator | None,
    ) -> list[EnsembleConfig]:
        """Generate TabPFN ensemble configs from training data."""
        if self.task == "classification":
            return EnsembleConfig.generate_for_classification(
                **self._ensemble_kwargs(X_train),
                class_shift_method=self.class_shift_method,
                n_classes=int(y_train.max()) + 1,
                random_state=random_state,
            )
        elif self.task == "regression":
            return EnsembleConfig.generate_for_regression(
                **self._ensemble_kwargs(X_train),
                target_transforms=self.target_transforms or [None],
                random_state=random_state,
            )
        else:
            raise ValueError(f"Invalid task: {self.task}")

    def _ensemble_kwargs(
        self,
        X_train: np.ndarray,
    ) -> dict:
        return {
            "n": self.n_estimators,
            "subsample_size": self.subsample_size,
            "add_fingerprint_feature": self.add_fingerprint_feature,
            "feature_shift_decoder": self.feature_shift_decoder,
            "polynomial_features": self.polynomial_features,
            "max_index": len(X_train),
            "augmentor_configs": self.augmentor_configs,
        }

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
        self.configs = self._generate_configs(
            X_train,
            y_train,
            random_state=random_state,
        )
        return super().fit(
            X_train,
            y_train,
            random_state=random_state,
            cat_ix=cat_ix,
        )
