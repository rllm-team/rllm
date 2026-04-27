"""Data augmentation utilities for the retained TabPFN runtime."""

from __future__ import annotations

from typing import Literal

import numpy as np

from rllm.data_augment.ensemble_augmentor import EnsembleAugmentor
from rllm.data_augment.ensemble_config import EnsembleConfig


class TabPFNEnsembleAugmentor(EnsembleAugmentor):
    """TabPFN augmentor made of multiple fitted ``AugmentorPipeline`` members."""

    def __init__(
        self,
        pipeline_configs: list[EnsembleConfig] | None = None,
        *,
        n_estimators: int = 4,
        task: Literal["classification", "regression"] = "classification",
        subsample_size: int | float | None = None,
        add_fingerprint_feature: bool = True,
        feature_shift_decoder: Literal["shuffle", "rotate"] | None = "shuffle",
        polynomial_features: Literal["no", "all"] | int = "no",
        class_shift_method: Literal["shuffle", "rotate"] | None = "shuffle",
        target_transforms: list | None = None,
        random_state: int | np.random.Generator | None = None,
        max_index: int | None = None,
        n_classes: int | None = None,
    ):
        self.task = task
        if pipeline_configs is None:
            pipeline_configs = [
                EnsembleConfig(
                    "quantile_uni",
                    append_original=False,
                    categorical_name="numeric",
                    global_transformer_name=None,
                    subsample_features=680,
                    transform_sequence=None,
                ),
                EnsembleConfig(
                    "quantile_uni",
                    append_original=False if task == "classification" else "auto",
                    categorical_name="ordinal_very_common_categories_shuffled",
                    global_transformer_name="svd_quarter_components",
                    subsample_features=500,
                    transform_sequence=None,
                ),
            ]

        self.task = task
        self.n_estimators = n_estimators
        self.subsample_size = subsample_size
        self.add_fingerprint_feature = add_fingerprint_feature
        self.feature_shift_decoder = feature_shift_decoder
        self.polynomial_features = polynomial_features
        self.class_shift_method = class_shift_method
        self.pipeline_configs = pipeline_configs
        self.target_transforms = target_transforms
        self.max_index = max_index
        self.n_classes = n_classes
        configs = self._generate_configs(random_state=random_state)
        super().__init__(configs, random_state=random_state)

    def _generate_configs(
        self,
        *,
        random_state: int | np.random.Generator | None = None,
    ) -> list[EnsembleConfig]:
        """Generate TabPFN ensemble configs."""
        if self.max_index is None:
            return []
        if self.task == "classification":
            if self.n_classes is None:
                raise ValueError("n_classes is required for classification configs.")
            return EnsembleConfig.generate_for_classification(
                n=self.n_estimators,
                subsample_size=self.subsample_size,
                add_fingerprint_feature=self.add_fingerprint_feature,
                feature_shift_decoder=self.feature_shift_decoder,
                polynomial_features=self.polynomial_features,
                max_index=self.max_index,
                pipeline_configs=self.pipeline_configs,
                class_shift_method=self.class_shift_method,
                n_classes=self.n_classes,
                random_state=random_state,
            )
        elif self.task == "regression":
            return EnsembleConfig.generate_for_regression(
                n=self.n_estimators,
                subsample_size=self.subsample_size,
                add_fingerprint_feature=self.add_fingerprint_feature,
                feature_shift_decoder=self.feature_shift_decoder,
                polynomial_features=self.polynomial_features,
                max_index=self.max_index,
                pipeline_configs=self.pipeline_configs,
                target_transforms=self.target_transforms or [None],
                random_state=random_state,
            )
        else:
            raise ValueError(f"Invalid task: {self.task}")
