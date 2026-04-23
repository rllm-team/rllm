from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np


@dataclass
class ClassifierPostProcessState:
    """Stores optional calibration parameters for inference-time postprocess."""

    temperature: float = 1.0
    threshold: float = 0.5


class ClassifierPostProcessor:
    """Applies optional threshold tuning and temperature scaling.

    Defaults are identity transforms so v2 behavior is preserved unless enabled.
    """

    def __init__(self) -> None:
        self.state = ClassifierPostProcessState()

    def fit_temperature(
        self,
        logits: np.ndarray,
        y_true: np.ndarray,
        *,
        max_iter: int = 30,
        lr: float = 0.05,
    ) -> None:
        if logits.ndim != 2 or logits.shape[1] <= 2:
            warnings.warn(
                "Temperature scaling currently targets multiclass logits; skipping.",
                stacklevel=2,
            )
            return
        y = y_true.astype(np.int64, copy=False).reshape(-1)
        t = np.array([self.state.temperature], dtype=np.float64)
        for _ in range(max_iter):
            probs = _stable_softmax(logits / np.clip(t, 1e-3, None))
            one_hot = np.zeros_like(probs)
            one_hot[np.arange(len(y)), y] = 1.0
            grad = np.sum((probs - one_hot) * logits) / max(1, logits.shape[0])
            t -= lr * grad
            t = np.clip(t, 1e-3, 100.0)
        self.state.temperature = float(t.item())

    def fit_threshold(self, prob_pos: np.ndarray, y_true: np.ndarray) -> None:
        y = y_true.astype(np.int64, copy=False).reshape(-1)
        if len(np.unique(y)) != 2:
            warnings.warn(
                "Threshold tuning currently supports binary classification only; skipping.",
                stacklevel=2,
            )
            return
        best_thr = 0.5
        best_score = -np.inf
        for thr in np.linspace(0.1, 0.9, 81):
            pred = (prob_pos >= thr).astype(np.int64)
            score = (pred == y).mean()
            if score > best_score:
                best_thr = float(thr)
                best_score = float(score)
        self.state.threshold = best_thr

    def apply_temperature(self, logits: np.ndarray) -> np.ndarray:
        if self.state.temperature == 1.0:
            return logits
        return logits / max(self.state.temperature, 1e-6)

    def apply_threshold(self, probs: np.ndarray) -> np.ndarray:
        if probs.ndim == 2 and probs.shape[1] == 2:
            adjusted = probs.copy()
            pred_pos = probs[:, 1] >= self.state.threshold
            adjusted[:, 1] = np.where(pred_pos, probs[:, 1], 1.0 - probs[:, 0])
            adjusted[:, 0] = 1.0 - adjusted[:, 1]
            return adjusted
        return probs

    def state_dict(self) -> dict[str, float]:
        return {
            "temperature": self.state.temperature,
            "threshold": self.state.threshold,
        }

    def load_state_dict(self, state: dict[str, float]) -> None:
        self.state.temperature = float(state.get("temperature", 1.0))
        self.state.threshold = float(state.get("threshold", 0.5))


def _stable_softmax(x: np.ndarray) -> np.ndarray:
    x_ = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x_)
    return e / np.sum(e, axis=1, keepdims=True)
