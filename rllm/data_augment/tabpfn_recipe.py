from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from rllm.data_augment.ensemble_augmentors import AugmentorConfig


@dataclass
class RecipeOptions:
    recipe: Literal["v2_6_default"] = "v2_6_default"
    use_robust_scaling: bool = False
    use_soft_clipping: bool = False
    use_quantile_transform: bool = False
    use_standard_scaling: bool = False
    keep_svd_augmentation: bool = True


def build_classifier_recipe_configs(options: RecipeOptions) -> list[AugmentorConfig]:
    transform_sequence = _build_transform_sequence(options)
    return [
        AugmentorConfig(
            "quantile_uni",
            append_original=False,
            categorical_name="numeric",
            global_transformer_name=None,
            subsample_features=680,
            transform_sequence=transform_sequence,
        ),
        AugmentorConfig(
            "quantile_uni",
            append_original=False,
            categorical_name="ordinal_very_common_categories_shuffled",
            global_transformer_name=(
                "svd_quarter_components"
                if options.keep_svd_augmentation
                else "scaler"
            ),
            subsample_features=500,
            transform_sequence=transform_sequence,
        ),
    ]


def build_regressor_recipe_configs(options: RecipeOptions) -> list[AugmentorConfig]:
    transform_sequence = _build_transform_sequence(options)
    return [
        AugmentorConfig(
            "quantile_uni",
            append_original=False,
            categorical_name="numeric",
            global_transformer_name=None,
            subsample_features=680,
            transform_sequence=transform_sequence,
        ),
        AugmentorConfig(
            "quantile_uni",
            append_original="auto",
            categorical_name="ordinal_very_common_categories_shuffled",
            global_transformer_name=(
                "svd_quarter_components"
                if options.keep_svd_augmentation
                else "scaler"
            ),
            subsample_features=500,
            transform_sequence=transform_sequence,
        ),
    ]


def _build_transform_sequence(options: RecipeOptions) -> tuple[str, ...] | None:
    seq: list[str] = []
    if options.use_robust_scaling:
        seq.append("robust")
    if options.use_soft_clipping:
        seq.append("soft_clip")
    if options.use_quantile_transform:
        seq.append("quantile_norm")
    if options.use_standard_scaling:
        seq.append("standard")
    return tuple(seq) if seq else None
