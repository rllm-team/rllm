#  Copyright (c) Prior Labs GmbH 2025.

"""Add fingerprint features based on hash of each row."""

from __future__ import annotations

import numpy as np
from typing_extensions import override

from rllm.data_augment.data_augmentor import DataAugmentor
from rllm.data_augment.preprocessing_utils import float_hash_arr
from rllm.data_augment.utils import infer_random_state


class AddFingerprintFeaturesAugmentor(DataAugmentor):
    """Adds a fingerprint feature to the features based on hash of each row.

    If `is_test = True`, it keeps the first hash even if there are collisions.
    If `is_test = False`, it handles hash collisions by counting up and rehashing
    until a unique hash is found.
    """

    def __init__(self, random_state: int | np.random.Generator | None = None):
        super().__init__()
        self.random_state = random_state

    @override
    def _fit(self, X: np.ndarray, categorical_features: list[int]) -> list[int]:
        _, rng = infer_random_state(self.random_state)
        self.rnd_salt_ = int(rng.integers(0, 2**16))
        return [*categorical_features]

    @override
    def _transform(self, X: np.ndarray, *, is_test: bool = False) -> np.ndarray:
        X_h = np.zeros(X.shape[0], dtype=X.dtype)

        if is_test:
            # Keep the first hash even if there are collisions
            salted_X = X + self.rnd_salt_
            for i, row in enumerate(salted_X):
                h = float_hash_arr(row + self.rnd_salt_)
                X_h[i] = h
        else:
            # Handle hash collisions by counting up and rehashing
            seen_hashes = set()
            salted_X = X + self.rnd_salt_
            for i, row in enumerate(salted_X):
                h = float_hash_arr(row)
                add_to_hash = 0
                while h in seen_hashes:
                    add_to_hash += 1
                    h = float_hash_arr(row + add_to_hash)
                X_h[i] = h
                seen_hashes.add(h)
        print(
            f"Added fingerprint feature with {len(set(X_h)),X_h.shape} unique values.{np.concatenate([X, X_h.reshape(-1, 1)], axis=1).shape}"
        )
        return np.concatenate([X, X_h.reshape(-1, 1)], axis=1)
