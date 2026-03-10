#  Copyright (c) Prior Labs GmbH 2025.

"""None transformer that does nothing."""

from __future__ import annotations

from sklearn.preprocessing import FunctionTransformer

from rllm.data_augment.utils import _identity


class NoneTransformer(FunctionTransformer):
    """Transformer that does nothing (identity function)."""

    def __init__(self) -> None:
        super().__init__(func=_identity, inverse_func=_identity, check_inverse=False)
