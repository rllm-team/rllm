#  Copyright (c) Prior Labs GmbH 2025.

"""Add fingerprint features based on hash of each row."""

from __future__ import annotations

import hashlib
from collections import defaultdict

import numpy as np
from typing_extensions import override

from rllm.data_augment.data_augmentor import DataAugmentor

_CONSTANT = 2**64 - 1
_MAX_COLLISION_RETRIES = 100
_HASH_ROUND_DECIMALS = 12


def _hash_row_bytes(row_data: bytes, salt_bytes: bytes) -> float:
    """Hash pre-rounded row bytes with salt and map to [0, 1]."""
    _hash = int(hashlib.sha256(row_data + salt_bytes).hexdigest(), 16)
    return (_hash & _CONSTANT) / _CONSTANT


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
        # Match the official retained TabPFN logic: use dataset size as a stable salt.
        self.n_cells_ = X.shape[0] * X.shape[1]
        return [*categorical_features]

    @override
    def _transform(self, X: np.ndarray, *, is_test: bool = False) -> np.ndarray:
        X_rounded = np.ascontiguousarray(np.around(X, decimals=_HASH_ROUND_DECIMALS))
        salt_bytes = int(self.n_cells_).to_bytes(8, "little", signed=False)
        X_h = np.zeros(X.shape[0], dtype=X.dtype)

        if is_test:
            for i in range(X_rounded.shape[0]):
                row_data = X_rounded[i].tobytes()
                X_h[i] = _hash_row_bytes(row_data, salt_bytes)
        else:
            seen_hashes = set()
            hash_counter: dict[float, int] = defaultdict(int)

            def _hash_with_offset(row_bytes: bytes, offset: int) -> float:
                offset_bytes = int(self.n_cells_ + offset).to_bytes(
                    8, "little", signed=False
                )
                return _hash_row_bytes(row_bytes, offset_bytes)

            for i in range(X_rounded.shape[0]):
                row_data = X_rounded[i].tobytes()
                h_base = _hash_row_bytes(row_data, salt_bytes)
                add_to_hash = hash_counter[h_base]
                h = h_base if add_to_hash == 0 else _hash_with_offset(row_data, add_to_hash)

                retries = 0
                while h in seen_hashes and not np.isnan(X[i]).all():
                    add_to_hash += 1
                    retries += 1
                    if retries > _MAX_COLLISION_RETRIES:
                        raise RuntimeError(
                            "Fingerprint hash collision not resolved after "
                            f"{_MAX_COLLISION_RETRIES} retries for row {i}."
                        )
                    h = _hash_with_offset(row_data, add_to_hash)
                X_h[i] = h
                seen_hashes.add(h)
                hash_counter[h_base] = add_to_hash + 1
        return np.concatenate([X, X_h.reshape(-1, 1)], axis=1)
