"""Defines the augmentation configurations that define the ensembling of
different members.
"""

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


@dataclass
class AugmentorConfig:
    """Configuration for data augmentors."""

    name: Literal[
        "per_feature",
        "power",
        "safepower",
        "power_box",
        "safepower_box",
        "quantile_uni_coarse",
        "quantile_norm_coarse",
        "quantile_uni",
        "quantile_norm",
        "quantile_uni_fine",
        "quantile_norm_fine",
        "robust",
        "kdi",
        "none",
        "kdi_random_alpha",
        "kdi_uni",
        "kdi_random_alpha_uni",
        "adaptive",
        "norm_and_kdi",
        "squashing_scaler_default",
        "kdi_alpha_0.3_uni",
        "kdi_alpha_0.5_uni",
        "kdi_alpha_0.8_uni",
        "kdi_alpha_1.0_uni",
        "kdi_alpha_1.2_uni",
        "kdi_alpha_1.5_uni",
        "kdi_alpha_2.0_uni",
        "kdi_alpha_3.0_uni",
        "kdi_alpha_5.0_uni",
        "kdi_alpha_0.3",
        "kdi_alpha_0.5",
        "kdi_alpha_0.8",
        "kdi_alpha_1.0",
        "kdi_alpha_1.2",
        "kdi_alpha_1.5",
        "kdi_alpha_2.0",
        "kdi_alpha_3.0",
        "kdi_alpha_5.0",
    ]
    categorical_name: Literal[
        "none",
        "numeric",
        "onehot",
        "ordinal",
        "ordinal_shuffled",
        "ordinal_very_common_categories_shuffled",
    ] = "none"
    append_original: bool | Literal["auto"] = False
    subsample_features: int | float = -1
    global_transformer_name: str | None = None
    transform_sequence: tuple[str, ...] | None = None

    @override
    def __str__(self) -> str:
        return (
            f"{self.name}_cat:{self.categorical_name}"
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


def generate_index_permutations(
    n: int,
    *,
    max_index: int,
    subsample: int | float,
    random_state: int | np.random.Generator | None,
) -> list[npt.NDArray[np.int64]]:
    """Generate indices for subsampling from the data."""
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
    """Configuration for an ensemble member."""

    augment_config: AugmentorConfig
    add_fingerprint_feature: bool
    polynomial_features: Literal["no", "all"] | int
    feature_shift_count: int
    feature_shift_decoder: Literal["shuffle", "rotate"] | None
    subsample_ix: npt.NDArray[np.int64] | None

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
        augmentor_configs: Sequence[AugmentorConfig],
        class_shift_method: Literal["rotate", "shuffle"] | None,
        n_classes: int,
        random_state: int | np.random.Generator | None,
    ) -> list[ClassifierEnsembleConfig]:
        """Generate ensemble configurations for classification."""
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

        balance_count = n // len(augmentor_configs)
        configs_ = balance(augmentor_configs, balance_count)
        leftover = n - len(configs_)
        if leftover > 0:
            configs_.extend(augmentor_configs[:leftover])

        return [
            ClassifierEnsembleConfig(
                augment_config=augment_config,
                feature_shift_count=featshift,
                add_fingerprint_feature=add_fingerprint_feature,
                polynomial_features=polynomial_features,
                feature_shift_decoder=feature_shift_decoder,
                subsample_ix=subsample_ix,
                class_permutation=class_perm,
            )
            for featshift, augment_config, subsample_ix, class_perm in zip(
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
        augmentor_configs: Sequence[AugmentorConfig],
        target_transforms: Sequence[TransformerMixin | Pipeline | None],
        random_state: int | np.random.Generator | None,
    ) -> list[RegressorEnsembleConfig]:
        """Generate ensemble configurations for regression."""
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

        combos = list(product(augmentor_configs, target_transforms))
        balance_count = n // len(combos)
        configs_ = balance(combos, balance_count)
        rand_count = n % len(combos)
        if rand_count > 0:
            configs_ += combos[:rand_count]

        return [
            RegressorEnsembleConfig(
                augment_config=augment_config,
                feature_shift_count=featshift,
                add_fingerprint_feature=add_fingerprint_feature,
                polynomial_features=polynomial_features,
                feature_shift_decoder=feature_shift_decoder,
                subsample_ix=subsample_ix,
                target_transform=target_transform,
            )
            for featshift, subsample_ix, (augment_config, target_transform) in zip(
                featshifts,
                subsamples,
                configs_,
            )
        ]


@dataclass
class ClassifierEnsembleConfig(EnsembleConfig):
    """Configuration for a classifier ensemble member."""

    class_permutation: np.ndarray | None


@dataclass
class RegressorEnsembleConfig(EnsembleConfig):
    """Configuration for a regression ensemble member."""

    target_transform: TransformerMixin | Pipeline | None
