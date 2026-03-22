from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from torch import Tensor

from rllm.types import ColType, StatType
from ._col_encoder import ColEncoder


class CategoricalInputPerFeatureEncoder(ColEncoder):
    r"""Per-feature categorical encoder inspired by TabPFN counterpart.

    This encoder can process mixed categorical/continuous columns. Column-level
    modality is inferred from metadata key ``"is_categorical"`` (defaults True).
    """

    supported_types = {ColType.CATEGORICAL, ColType.BINARY}

    def __init__(
        self,
        num_embs: int = 1000,
        out_dim: Optional[int] = None,
        stats_list: Optional[List[Dict[StatType, Any]]] = None,
        post_module: Optional[torch.nn.Module] = None,
    ) -> None:
        super().__init__(
            out_dim=out_dim, stats_list=stats_list, post_module=post_module
        )
        self.num_embs = num_embs

    def post_init(self) -> None:
        if self.out_dim is None:
            raise ValueError(
                "out_dim must be set for CategoricalInputPerFeatureEncoder"
            )
        self.embedding = torch.nn.Embedding(self.num_embs, self.out_dim)
        self.cont_proj = torch.nn.Linear(1, self.out_dim)

        is_categorical: list[bool] = []
        for stats in self.stats_list:
            is_categorical.append(bool(stats.get("is_categorical", True)))
        self.register_buffer(
            "is_categorical_mask", torch.tensor(is_categorical, dtype=torch.bool)
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        super().reset_parameters()
        self.embedding.reset_parameters()
        self.cont_proj.reset_parameters()

    def encode_forward(self, feat: Tensor) -> Tensor:
        if feat.ndim == 3 and feat.shape[-1] == 1:
            x = feat.squeeze(-1)
        elif feat.ndim == 2:
            x = feat
        else:
            raise ValueError(
                f"Expected feat shape [B,C] or [B,C,1], got {tuple(feat.shape)}"
            )

        if x.shape[1] != self.is_categorical_mask.numel():
            raise ValueError(
                "Feature width does not match metadata width in "
                "CategoricalInputPerFeatureEncoder"
            )

        cat_mask = self.is_categorical_mask.to(x.device)
        out = torch.zeros(x.shape[0], x.shape[1], self.out_dim, device=x.device)

        if cat_mask.any():
            x_cat = x[:, cat_mask]
            nan_mask = torch.isnan(x_cat) | torch.isinf(x_cat)
            x_cat = x_cat.long().clamp(0, self.num_embs - 2)
            x_cat[nan_mask] = self.num_embs - 1
            out[:, cat_mask] = self.embedding(x_cat)

        if (~cat_mask).any():
            x_cont = x[:, ~cat_mask].unsqueeze(-1)
            out[:, ~cat_mask] = self.cont_proj(x_cont)

        if self.post_module is not None:
            out = self.post_module(out)

        return out
