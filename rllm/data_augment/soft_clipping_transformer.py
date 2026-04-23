from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class SoftClippingTransformer(BaseEstimator, TransformerMixin):
    """Smoothly clip heavy tails with tanh scaling.

    This keeps monotonicity while reducing outlier magnitude.
    """

    def __init__(self, clip_value: float = 5.0) -> None:
        self.clip_value = float(clip_value)

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "SoftClippingTransformer":
        del y
        self.scale_ = np.nanstd(X, axis=0, keepdims=True)
        self.scale_ = np.where(self.scale_ <= 1e-8, 1.0, self.scale_)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "scale_"):
            raise RuntimeError("SoftClippingTransformer must be fitted before transform.")
        z = X / self.scale_
        return np.tanh(z / self.clip_value) * self.clip_value * self.scale_
