from __future__ import annotations
from typing import Literal

import torch
import torch.nn.functional as F

from rllm.nn.loss.base_loss import BaseLoss


class ContrastiveLoss(BaseLoss):
    r"""
    Generalized InfoNCE-style contrastive loss with a customizable positive mask.

    This class provides a reusable implementation of the InfoNCE / SupCon
    contrastive objective used in self-supervised, supervised, and vertical-partition
    contrastive learning. Subclasses only define how positives are selected
    (via `pos_mask`); all numerical and normalization steps are handled here.

    Per-anchor loss:
    \[
        \ell_i = -\frac{1}{|P(i)|}
        \sum_{p \in P(i)} \log
        \frac{\exp(s_{ip} / \tau)}
             {\sum_{a \neq i} \exp(s_{ia} / \tau)}
    \]

    Batch loss (with scaling factor \(\frac{\tau}{\tau_0}\)):
    \[
        \mathcal{L} =
        \frac{\tau}{\tau_0} \cdot
        \frac{1}{N} \sum_{i=1}^{N} \ell_i
    \]

    Args:
        temperature (float): Temperature \(\tau\) scaling the logits.
        base_temperature (float): Reference temperature \(\tau_0\) for scaling.
        similarity (str): Similarity metric, "dot" or "cosine".
        eps (float): Numerical stability constant.
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
