from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass
class ExportArtifacts:
    """Open-source export placeholders for deployment pathways."""

    mode: str
    payload: dict


class DistillationMLP(nn.Module):
    """Lightweight student used by the MLP export stub."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def export_as_mlp_stub(
    X: np.ndarray,
    soft_targets: np.ndarray,
    *,
    hidden_dim: int = 128,
    n_steps: int = 50,
    lr: float = 1e-3,
) -> ExportArtifacts:
    """Train a tiny MLP on soft targets as an OSS-compatible placeholder."""
    X_tensor = torch.as_tensor(X, dtype=torch.float32)
    y_tensor = torch.as_tensor(soft_targets, dtype=torch.float32)
    model = DistillationMLP(
        in_dim=X_tensor.shape[1],
        out_dim=y_tensor.shape[1] if y_tensor.ndim == 2 else 1,
        hidden_dim=hidden_dim,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    model.train()
    for _ in range(n_steps):
        optimizer.zero_grad()
        pred = model(X_tensor)
        if y_tensor.ndim == 1:
            y_used = y_tensor.unsqueeze(-1)
        else:
            y_used = y_tensor
        loss = criterion(pred, y_used)
        loss.backward()
        optimizer.step()
    return ExportArtifacts(
        mode="mlp_stub",
        payload={
            "state_dict": model.state_dict(),
            "in_dim": X_tensor.shape[1],
            "out_dim": int(y_used.shape[1]),
            "hidden_dim": hidden_dim,
            "n_steps": n_steps,
        },
    )


def export_as_tree_ensemble_stub(*_: object, **__: object) -> ExportArtifacts:
    raise NotImplementedError(
        "Tree ensemble export is provided as interface only. "
        "This repository does not include a proprietary TabPFN distillation engine."
    )
