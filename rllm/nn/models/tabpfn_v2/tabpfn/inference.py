"""Module that defines the inference engine for TabPFN.

This version keeps a single, low-memory inference mode (`InferenceEngineOnDemand`).
More aggressive caching strategies have been removed to reduce complexity and
centralize preprocessing logic in higher-level callers or examples.
"""

#  Copyright (c) Prior Labs GmbH 2025.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing_extensions import override

import numpy as np
import torch
from rllm.data_augment.ensemble_preprocessing import fit_preprocessing

if TYPE_CHECKING:
    from .model.transformer import PerFeatureTransformer
    from rllm.data_augment.ensemble_preprocessing import EnsembleConfig


@dataclass
class InferenceEngine(ABC):
    """These define how tabpfn inference can be run.

    As there are many things that can be cached, with multiple ways to parallelize,
    `Executor` defines three primary things:

    Most will define a method `prepare()` which is specific to that inference engine.
    These do not share a common interface.

    1. What to cache:

        As we can prepare a lot of the transformers context, there is a tradeoff in
        terms of how much memory to be spent in caching. This memory is used when
        `prepare()` is called, usually in `fit()`.

    2. Using the cached data for inference:

        Based on what has been prepared for the transformer context,
        `iter_outputs()` will use this cached information to make predictions.

    3. Controlling parallelism:

        As we have trivially parallel parts for inference, we can parallelize them.
        However as the GPU is typically a bottle-neck in most systems, we can define,
        where and how we would like to parallelize the inference.
    """

    dtype_byte_size: int

    @abstractmethod
    def iter_outputs(
        self,
        X: np.ndarray,
        *,
        device: torch.device,
        autocast: bool,
    ) -> Iterator[tuple[torch.Tensor, EnsembleConfig]]:
        """Iterate over the outputs of the model.

        One for each ensemble configuration that was used to initialize the executor.

        Args:
            X: The input data to make predictions on.
            device: The device to run the model on.
            autocast: Whether to use torch.autocast during inference.
        """
        ...


@dataclass
class InferenceEngineOnDemand(InferenceEngine):
    """Inference engine that does not cache anything, computes everything as needed.

    This is one of the slowest ways to run inference, as computation that could be
    cached is recomputed on every call. However the memory demand is lowest and
    can be more trivially parallelized across GPUs with some work.
    """

    X_train: np.ndarray
    y_train: np.ndarray
    ensemble_configs: Sequence[EnsembleConfig]
    cat_ix: list[int]
    static_seed: int
    n_workers: int
    model: PerFeatureTransformer
    force_inference_dtype: torch.dtype | None

    @classmethod
    def prepare(
        cls,
        X_train: np.ndarray,
        y_train: np.ndarray,
        *,
        cat_ix: list[int],
        model: PerFeatureTransformer,
        ensemble_configs: Sequence[EnsembleConfig],
        rng: np.random.Generator,
        n_workers: int,
        dtype_byte_size: int,
        force_inference_dtype: torch.dtype | None,
    ) -> InferenceEngineOnDemand:
        """Prepare the inference engine.

        Args:
            X_train: The training data.
            y_train: The training target.
            cat_ix: The categorical indices.
            model: The model to use.
            ensemble_configs: The ensemble configurations to use.
            rng: The random number generator.
            n_workers: The number of workers to use.
            dtype_byte_size: The byte size of the dtype.
            force_inference_dtype: The dtype to force inference to.
        """
        # We save it as a static seed to be reproducible across predicts
        static_seed = rng.integers(0, 2**31)
        return cls(
            X_train=X_train,
            y_train=y_train,
            ensemble_configs=ensemble_configs,
            cat_ix=cat_ix,
            model=model,
            static_seed=static_seed,
            n_workers=n_workers,
            dtype_byte_size=dtype_byte_size,
            force_inference_dtype=force_inference_dtype,
        )

    @override
    def iter_outputs(
        self,
        X: np.ndarray,
        *,
        device: torch.device,
        autocast: bool,
    ) -> Iterator[tuple[torch.Tensor, EnsembleConfig]]:
        rng = np.random.default_rng(self.static_seed)
        itr = fit_preprocessing(
            configs=self.ensemble_configs,
            X_train=self.X_train,
            y_train=self.y_train,
            random_state=rng,
            cat_ix=self.cat_ix,
            n_workers=self.n_workers,
            parallel_mode="as-ready",
        )

        self.model = self.model.to(device)
        if self.force_inference_dtype is not None:
            self.model = self.model.type(self.force_inference_dtype)

        for config, preprocessor, X_train, y_train, cat_ix in itr:
            X_train = torch.as_tensor(
                X_train, dtype=torch.float32, device=device
            )  # noqa: PLW2901

            X_test = preprocessor.transform(X).X
            X_test = torch.as_tensor(X_test, dtype=torch.float32, device=device)

            X_full = torch.cat([X_train, X_test], dim=0).unsqueeze(1)
            y_train = torch.as_tensor(y_train, dtype=torch.float32, device=device)  # type: ignore  # noqa: PLW2901

            if self.force_inference_dtype is not None:
                X_full = X_full.type(self.force_inference_dtype)
                y_train = y_train.type(self.force_inference_dtype)  # type: ignore  # noqa: PLW2901

            style = None

            with (
                torch.autocast(device.type, enabled=autocast),
                torch.inference_mode(),
            ):
                output = self.model(
                    *(style, X_full, y_train),
                    only_return_standard_out=True,
                    categorical_inds=cat_ix,
                    single_eval_pos=len(y_train),
                )
            yield output.squeeze(1), config

        self.model = self.model.cpu()
