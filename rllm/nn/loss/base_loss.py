from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal


class BaseLoss(nn.Module):
    r"""
    BaseLoss

    Minimal root class for all custom loss functions in this repository.

    Rationale:
    - We want a single, consistent parent type so that higher-level code
      (trainer loops, registries, logging utilities) can treat every
      project-specific loss in a uniform way.
    - Concrete subclasses must implement `forward(...)` and return a
      scalar tensor.

    This class itself does not impose any particular training logic.
    It only standardizes the interface.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement forward()"
        )


class BaseContrastiveLoss(BaseLoss):
    r"""
    BaseContrastiveLoss

    A reusable InfoNCE / supervised-contrastive style loss core.

    This class encapsulates the *generic* contrastive computation pattern
    that appears in:
      - Self-supervised contrastive learning (InfoNCE / NT-Xent),
      - Supervised contrastive learning (SupCon),
      - Vertical-partition contrastive learning (VPCL) variants used in
        table representation learning (e.g. TransTab-style Self-VPCL /
        Supervised-VPCL).

    Key idea:
    Subclasses DO NOT re-implement the log-softmax math. They ONLY define
    what counts as a "positive pair" by building a positive-mask `pos_mask`.

    Notation:
        feats: [N, D]
            N = total number of contrastive instances (e.g. batch_size * num_partitions)
            D = projection dim
        pos_mask: [N, N]
            pos_mask[a, b] = 1.0 iff b is considered a positive sample for anchor a.
            pos_mask[a, a] can be 1.0 or 0.0 when provided; we will internally
            mask out exact self-pairs so they do not contribute.

    The loss matches your TransTab-style implementation:
        1. Compute pairwise similarity (dot or cosine),
        2. Divide by temperature,
        3. Row-wise max subtraction for numerical stability,
        4. Exclude self-contrast from denominator,
        5. For each anchor, compute average log_prob over its positives,
        6. Multiply by -(T / T0) and average over anchors.

    Args:
        temperature (float): τ in InfoNCE. Scales the logits.
        base_temperature (float): τ₀ for final scaling (T / T0).
        similarity (str): "dot" or "cosine".
            - "dot": raw dot product.
            - "cosine": L2-normalize feats first, then dot product (cosine sim).
        eps (float): numerical stability in logs and divisions.
    """

    def __init__(
        self,
        temperature: float = 10.0,
        base_temperature: float = 10.0,
        similarity: Literal["dot", "cosine"] = "dot",
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.temperature = float(temperature)
        self.base_temperature = float(base_temperature)
        self.similarity = similarity
        self.eps = eps

    def _pairwise_logits(self, feats: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise similarity logits BEFORE numerical stabilization.

        feats: [N, D]
        returns: [N, N] logits where logits[a, b] = sim(a, b) / temperature
        """
        if self.similarity == "cosine":
            feats = F.normalize(feats, dim=1)

        # dot product similarity matrix
        # [N, D] @ [D, N] -> [N, N]
        sim_matrix = torch.matmul(feats, feats.T)  # cosine if normalized, otherwise dot
        logits = sim_matrix / self.temperature
        return logits

    def forward(
        self,
        feats: torch.Tensor,
        pos_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the final contrastive loss given embeddings and a positive mask.

        Args:
            feats (Tensor): [N, D]
                Projected embeddings from all "views"/"partitions".
            pos_mask (Tensor): [N, N], dtype float/bool
                pos_mask[a, b] = 1.0 if sample b should be treated
                as a positive of anchor a. We will internally remove
                exact self-pairs from contributing positives and
                denominators.

        Returns:
            loss (Tensor): scalar contrastive loss.
        """
        device = feats.device
        N = feats.shape[0]

        # 1. Compute pairwise logits
        anchor_dot_contrast = self._pairwise_logits(feats)  # [N, N]

        # 2. Numerical stability: subtract row-wise max
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()  # [N, N]

        # 3. Build mask that excludes self-pairs from denominator
        # logits_mask[a, a] = 0, else 1
        logits_mask = torch.ones_like(pos_mask, dtype=feats.dtype, device=device)
        diag_idx = torch.arange(N, device=device).view(-1, 1)
        logits_mask.scatter_(1, diag_idx, 0.0)

        # 4. Positive mask should also not include self
        mask = (pos_mask.float() * logits_mask).to(feats.dtype)  # [N, N]

        # 5. Denominator for softmax: all non-self pairs
        exp_logits = torch.exp(logits) * logits_mask  # [N, N]
        denom = exp_logits.sum(dim=1, keepdim=True) + self.eps  # [N, 1]

        # log_prob[a, b] = log( p(b | a) ), where denominator excludes self
        log_prob = logits - torch.log(denom)  # [N, N] broadcast row-wise

        # 6. For each anchor a, average log_prob over its positives
        pos_weight_sum = mask.sum(dim=1)  # [N]
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (pos_weight_sum + self.eps)  # [N]

        # 7. Final scaling (matches your TransTab code):
        # loss_a = - (T / T0) * mean_log_prob_pos[a]
        loss_per_anchor = - (self.temperature / self.base_temperature) * mean_log_prob_pos  # [N]

        loss = loss_per_anchor.mean()
        return loss
