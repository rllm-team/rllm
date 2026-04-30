"""Defines augmentation configurations for TabPFN ensemble members."""

#  Copyright (c) Prior Labs GmbH 2025.

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import product
from typing import TYPE_CHECKING, Literal

import numpy as np
from typing_extensions import override

from rllm.data_augment.utils import balance, infer_random_state

if TYPE_CHECKING:
    import numpy.typing as npt
    from sklearn.base import TransformerMixin
    from sklearn.pipeline import Pipeline


AugmentorName = Literal[
    "quantile_uni_coarse",
    "quantile_norm_coarse",
    "quantile_uni",
    "quantile_norm",
    "quantile_uni_fine",
    "quantile_norm_fine",
    "none",
]
CategoricalAugmentorName = Literal[
    "none",
    "numeric",
    "onehot",
    "ordinal",
    "ordinal_shuffled",
    "ordinal_very_common_categories_shuffled",
]


def generate_index_permutations(
    n: int,
    *,
    max_index: int,
    subsample: int | float,
    random_state: int | np.random.Generator | None,
) -> list[npt.NDArray[np.int64]]:
    """Generate indices for row subsampling."""
    _, rng = infer_random_state(random_state)
    if isinstance(subsample, int):
        if not (1 <= subsample <= max_index):
            raise ValueError(f"{subsample=} must be in [1, {max_index}] if int")
        return [rng.permutation(max_index)[:subsample] for _ in range(n)]

    if isinstance(subsample, float):
        if not (0 < subsample < 1):
            raise ValueError(f"{subsample=} must be in (0, 1) if float")
        subsample = int(subsample * max_index) + 1
        return [rng.permutation(max_index)[:subsample] for _ in range(n)]

    raise ValueError(f"{subsample=} must be int or float.")


@dataclass
class EnsembleConfig:
    """Configuration for one ensemble member and its augmentation pipeline."""

    augmentor: AugmentorName
    categorical_name: CategoricalAugmentorName = "none"
    append_original: bool | Literal["auto"] = False
    subsample_features: int | float = -1
    global_transformer_name: str | None = None
    transform_sequence: tuple[str, ...] | None = None
    add_fingerprint_feature: bool = True
    polynomial_features: Literal["no", "all"] | int = "no"
    feature_shift_count: int = 0
    feature_shift_decoder: Literal["shuffle", "rotate"] | None = "shuffle"
    subsample_ix: npt.NDArray[np.int64] | None = None

    @override
    def __str__(self) -> str:
        return (
            f"{self.augmentor}_cat:{self.categorical_name}"
            + ("_and_none" if self.append_original else "")
            + (
                f"_subsample_feats_{self.subsample_features}"
                if self.subsample_features > 0
                else ""
            )
            + (
                f"_global_transformer_{self.global_transformer_name}"
                if self.global_transformer_name is not None
                else ""
            )
            + (
                f"_seq_{'-'.join(self.transform_sequence)}"
                if self.transform_sequence is not None
                else ""
            )
        )

    def pipeline_kwargs(self) -> dict:
        return {
            "augmentor": self.augmentor,
            "categorical_name": self.categorical_name,
            "append_original": self.append_original,
            "subsample_features": self.subsample_features,
            "global_transformer_name": self.global_transformer_name,
            "transform_sequence": self.transform_sequence,
        }

    @classmethod
    def generate_for_classification(
        cls,
        *,
        n: int,
        subsample_size: int | float | None,
        max_index: int,
        add_fingerprint_feature: bool,
        polynomial_features: Literal["no", "all"] | int,
        feature_shift_decoder: Literal["shuffle", "rotate"] | None,
        pipeline_configs: Sequence[EnsembleConfig],
        class_shift_method: Literal["rotate", "shuffle"] | None,
        n_classes: int,
        random_state: int | np.random.Generator | None,
    ) -> list[ClassifierEnsembleConfig]:
        """Generate one config per classifier ensemble member."""
        static_seed, rng = infer_random_state(random_state)
        start = rng.integers(0, 1000)
        featshifts = np.arange(start, start + n)
        featshifts = rng.choice(featshifts, size=n, replace=False)  # type: ignore

        if class_shift_method == "rotate":
            arange = np.arange(0, n_classes)
            shifts = rng.permutation(n_classes).tolist()
            class_permutations = [np.roll(arange, s) for s in shifts]
            class_permutations = [  # type: ignore
                class_permutations[c] for c in rng.choice(n_classes, n)
            ]
        elif class_shift_method == "shuffle":
            noise = rng.random((n * 3, n_classes))
            shufflings = np.argsort(noise, axis=1)
            uniqs = np.unique(shufflings, axis=0)
            balance_count = n // len(uniqs)
            class_permutations = balance(uniqs, balance_count)
            rand_count = n % len(uniqs)
            if rand_count > 0:
                class_permutations += [  # type: ignore
                    uniqs[i] for i in rng.choice(len(uniqs), size=rand_count)
                ]
        elif class_shift_method is None:
            class_permutations = [None] * n  # type: ignore
        else:
            raise ValueError(f"Unknown {class_shift_method=}")

        if isinstance(subsample_size, (int, float)):
            subsamples: list[None] | list[np.ndarray] = generate_index_permutations(
                n=n,
                max_index=max_index,
                subsample=subsample_size,
                random_state=static_seed,
            )
        elif subsample_size is None:
            subsamples = [None] * n  # type: ignore
        else:
            raise ValueError(f"Invalid subsample_samples: {subsample_size}")

        balance_count = n // len(pipeline_configs)
        configs_ = balance(pipeline_configs, balance_count)
        leftover = n - len(configs_)
        if leftover > 0:
            configs_.extend(pipeline_configs[:leftover])

        return [
            ClassifierEnsembleConfig(
                **pipeline_config.pipeline_kwargs(),
                feature_shift_count=featshift,
                add_fingerprint_feature=add_fingerprint_feature,
                polynomial_features=polynomial_features,
                feature_shift_decoder=feature_shift_decoder,
                subsample_ix=subsample_ix,
                class_permutation=class_perm,
            )
            for featshift, pipeline_config, subsample_ix, class_perm in zip(
                featshifts,
                configs_,
                subsamples,
                class_permutations,
            )
        ]

    @classmethod
    def generate_for_regression(
        cls,
        *,
        n: int,
        subsample_size: int | float | None,
        max_index: int,
        add_fingerprint_feature: bool,
        polynomial_features: Literal["no", "all"] | int,
        feature_shift_decoder: Literal["shuffle", "rotate"] | None,
        pipeline_configs: Sequence[EnsembleConfig],
        target_transforms: Sequence[TransformerMixin | Pipeline | None],
        random_state: int | np.random.Generator | None,
    ) -> list[RegressorEnsembleConfig]:
        """Generate one config per regression ensemble member."""
        static_seed, rng = infer_random_state(random_state)
        start = rng.integers(0, 1000)
        featshifts = np.arange(start, start + n)
        featshifts = rng.choice(featshifts, size=n, replace=False)  # type: ignore

        if isinstance(subsample_size, (int, float)):
            subsamples: list[None] | list[np.ndarray] = generate_index_permutations(
                n=n,
                max_index=max_index,
                subsample=subsample_size,
                random_state=static_seed,
            )
        elif subsample_size is None:
            subsamples = [None] * n
        else:
            raise ValueError(f"Invalid subsample_samples: {subsample_size}")

        combos = list(product(pipeline_configs, target_transforms))
        balance_count = n // len(combos)
        configs_ = balance(combos, balance_count)
        rand_count = n % len(combos)
        if rand_count > 0:
            configs_ += combos[:rand_count]

        return [
            RegressorEnsembleConfig(
                **pipeline_config.pipeline_kwargs(),
                feature_shift_count=featshift,
                add_fingerprint_feature=add_fingerprint_feature,
                polynomial_features=polynomial_features,
                feature_shift_decoder=feature_shift_decoder,
                subsample_ix=subsample_ix,
                target_transform=target_transform,
            )
            for featshift, subsample_ix, (pipeline_config, target_transform) in zip(
                featshifts,
                subsamples,
                configs_,
            )
        ]


@dataclass
class ClassifierEnsembleConfig(EnsembleConfig):
    """Configuration for a classifier ensemble member."""

    class_permutation: np.ndarray | None = None


@dataclass
class RegressorEnsembleConfig(EnsembleConfig):
    """Configuration for a regression ensemble member."""

    target_transform: TransformerMixin | Pipeline | None = None
