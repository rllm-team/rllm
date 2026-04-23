from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.preprocessing import QuantileTransformer


class AdaptiveQuantileTransformer(QuantileTransformer):
    """QuantileTransformer with fit-time n_quantiles adaptation.

    This mirrors the official TabPFN behavior and avoids requesting more
    quantiles than are practical for the available sample count.
    """

    def __init__(
        self,
        *,
        n_quantiles: int = 1_000,
        subsample: int = 100_000,
        **kwargs: Any,
    ) -> None:
        self._user_n_quantiles = n_quantiles
        super().__init__(n_quantiles=n_quantiles, subsample=subsample, **kwargs)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
    ) -> "AdaptiveQuantileTransformer":
        n_samples = X.shape[0]
        self.n_quantiles = max(
            1,
            min(
                self._user_n_quantiles,
                n_samples,
                int(self.subsample * 0.2),
            ),
        )

        if isinstance(self.random_state, np.random.Generator):
            seed = int(self.random_state.integers(0, 2**32))
            self.random_state = np.random.RandomState(seed)
        elif hasattr(self.random_state, "bit_generator"):
            raise ValueError(
                f"Unsupported random state type: {type(self.random_state)}. "
                "Please provide an integer seed or np.random.RandomState object."
            )

        return super().fit(X, y)
