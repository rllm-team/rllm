from __future__ import annotations

from pathlib import Path

import torch

from rllm.nn.encoder.col_encoder._input_normalization_encoder import (
    torch_nanmean,
    torch_nanstd,
)

PREGENERATED_COLUMN_EMBEDDINGS_FILENAME = "pre_generated_column_embeddings_v2_6.pt"


def load_column_embeddings(path: str | Path) -> torch.Tensor:
    """Load persisted TabPFN column embeddings from the checkpoint directory."""

    col_embedding_path = Path(path)
    return torch.load(col_embedding_path, map_location="cpu", weights_only=True)


class TorchStandardScaler:
    """Torch implementation of standard scaling with NaN-aware statistics."""

    def fit(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        mean = torch_nanmean(x, axis=0)
        std = torch_nanstd(x, axis=0)

        std = torch.where(std == 0, torch.ones_like(std), std)
        if x.shape[0] == 1:
            std = torch.ones_like(std)

        return {"mean": mean, "std": std}

    def transform(
        self,
        x: torch.Tensor,
        fitted_cache: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if "mean" not in fitted_cache or "std" not in fitted_cache:
            raise ValueError("Invalid fitted cache. Must contain 'mean' and 'std'.")

        mean = fitted_cache["mean"]
        std = fitted_cache["std"]
        x = (x - mean) / (std + torch.finfo(std.dtype).eps)
        return torch.clip(x, min=-100, max=100)

    def __call__(
        self,
        x: torch.Tensor,
        num_train_rows: int | None = None,
    ) -> torch.Tensor:
        if num_train_rows is not None and num_train_rows > 0:
            fit_data = x[:num_train_rows]
        else:
            fit_data = x

        fitted_cache = self.fit(fit_data)
        return self.transform(x, fitted_cache=fitted_cache)
