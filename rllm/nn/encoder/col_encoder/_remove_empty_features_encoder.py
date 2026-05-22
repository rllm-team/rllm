from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from torch import Tensor

from rllm.types import ColType, StatType
from ._col_encoder import ColEncoder


def select_features(x: torch.Tensor, sel: torch.Tensor) -> torch.Tensor:
    B, total_features = sel.shape
    sequence_length = x.shape[0]

    if B == 1:
        return x[:, :, sel[0]]

    new_x = torch.zeros(
        (sequence_length, B, total_features),
        device=x.device,
        dtype=x.dtype,
    )

    sel_counts = sel.sum(dim=-1)

    for b in range(B):
        s = int(sel_counts[b])
        if s > 0:
            new_x[:, b, :s] = x[:, b, sel[b]]

    return new_x


class RemoveEmptyFeaturesEncoder(ColEncoder):
    r"""A ColEncoder that removes constant columns across the current batch.

    This encoder keeps the output width stable by moving non-empty columns to
    the left and padding the remaining columns with zeros.

    Args:
        out_dim (int, optional): Unused for this encoder.
        stats_list (List[Dict[StatType, Any]], optional): Optional metadata.
        post_module (torch.nn.Module, optional): Optional post module.
    """

    supported_types = {
        ColType.NUMERICAL,
        ColType.CATEGORICAL,
        ColType.BINARY,
    }

    def __init__(
        self,
        out_dim: Optional[int] = None,
        stats_list: Optional[List[Dict[StatType, Any]]] = None,
        post_module: Optional[torch.nn.Module] = None,
    ) -> None:
        super().__init__(
            out_dim=out_dim, stats_list=stats_list, post_module=post_module
        )

    def post_init(self) -> None:
        # No trainable parameters or required precomputed statistics.
        return None

    def reset_parameters(self) -> None:
        super().reset_parameters()

    def encode_forward(
        self,
        feat: Tensor,
        *,
        single_eval_pos: Optional[int] = None,
        normalize_on_train_only: bool = True,
    ) -> Tensor:
        if feat.ndim not in (2, 3):
            raise ValueError(
                f"Expected feat to be 2D or 3D, but got shape {tuple(feat.shape)}."
            )

        if feat.ndim == 2:
            # [batch_size, num_cols]
            ref = feat[0:1, :]
            sel = (feat != ref).any(dim=0)
            kept = feat[:, sel]
            pad_cols = feat.shape[1] - kept.shape[1]
            if pad_cols > 0:
                zeros = torch.zeros(
                    feat.shape[0], pad_cols, device=feat.device, dtype=feat.dtype
                )
                feat_out = torch.cat([kept, zeros], dim=1)
            else:
                feat_out = kept
        elif single_eval_pos is not None:
            train_x = feat[:single_eval_pos]
            sel = (train_x[1:] == train_x[0]).sum(0) != (train_x.shape[0] - 1)
            feat_out = select_features(feat, sel)
        else:
            # [batch_size, num_cols, emb_dim]
            ref = feat[0:1, :, :]
            sel = (feat != ref).any(dim=0).any(dim=-1)
            kept = feat[:, sel, :]
            pad_cols = feat.shape[1] - kept.shape[1]
            if pad_cols > 0:
                zeros = torch.zeros(
                    feat.shape[0],
                    pad_cols,
                    feat.shape[2],
                    device=feat.device,
                    dtype=feat.dtype,
                )
                feat_out = torch.cat([kept, zeros], dim=1)
            else:
                feat_out = kept

        if self.post_module is not None:
            feat_out = self.post_module(feat_out)

        return feat_out
