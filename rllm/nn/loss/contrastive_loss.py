from __future__ import annotations
from typing import Literal

import torch
import torch.nn.functional as F

from rllm.nn.loss.base_loss import BaseLoss


class ContrastiveLoss(BaseLoss):
    r"""Generalized InfoNCE-style contrastive loss with a customizable positive mask.

    This class provides a reusable implementation of the InfoNCE / SupCon
    contrastive objective used in self-supervised, supervised, and
    vertical-partition contrastive learning.  Subclasses only define how
    positives are selected (via ``pos_mask``); all numerical and
    normalisation steps are handled here.

    Per-anchor loss:

    .. math::

        \ell_i = -\frac{1}{|P(i)|}
        \sum_{p \in P(i)} \log
        \frac{\exp(s_{ip} / \tau)}
             {\sum_{a \neq i} \exp(s_{ia} / \tau)}

    Batch loss (with scaling factor :math:`\frac{\tau}{\tau_0}`):

    .. math::

        \mathcal{L} =
        \frac{\tau}{\tau_0} \cdot
        \frac{1}{N} \sum_{i=1}^{N} \ell_i

    Args:
        temperature (float): Temperature :math:`\tau` scaling the logits.
        base_temperature (float): Reference temperature :math:`\tau_0`.
        similarity (str): Similarity metric, ``"dot"`` or ``"cosine"``.
        eps (float): Numerical stability constant added to log-denominators.

    Example:
        >>> import torch
        >>> loss_fn = ContrastiveLoss(temperature=1.0, similarity="dot")
        >>> feats = torch.randn(4, 8)
        >>> pos_mask = torch.eye(4)
        >>> loss_fn(feats, pos_mask).ndim
        0
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
        """Compute pairwise similarity logits scaled by temperature.

        Args:
            feats: ``[N, D]`` embedding matrix.

        Returns:
            ``[N, N]`` logit matrix where ``logits[a, b] = sim(a, b) / τ``.
        """
        if self.similarity == "cosine":
            feats = F.normalize(feats, dim=1)
        sim_matrix = torch.matmul(feats, feats.T)
        return sim_matrix / self.temperature

    def forward(
        self,
        feats: torch.Tensor,
        pos_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the contrastive loss given embeddings and a positive mask.

        Args:
            feats (Tensor): ``[N, D]`` projected embeddings from all views /
                partitions.
            pos_mask (Tensor): ``[N, N]`` float or bool mask where
                ``pos_mask[a, b] = 1`` when sample *b* should be treated as a
                positive of anchor *a*.  Self-pairs are excluded internally.

        Returns:
            torch.Tensor: Scalar contrastive loss.  Returns ``0.0`` (with
            gradient) when no anchor in the batch has any valid positive.

        Example:
            >>> import torch
            >>> loss_fn = ContrastiveLoss()
            >>> feats = torch.randn(3, 5)
            >>> pos_mask = torch.tensor(
            ...     [[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=torch.float32
            ... )
            >>> loss_fn(feats, pos_mask).ndim
            0
        """
        device = feats.device
        N = feats.shape[0]

        # 1. Pairwise logits [N, N]
        logits = self._pairwise_logits(feats)

        # 2. Set diagonal to -inf so self-pairs contribute exactly 0 in exp
        #    without the nan-producing `exp(large) * 0` pattern.
        eye = torch.eye(N, dtype=torch.bool, device=device)
        logits = logits.masked_fill(eye, float("-inf"))

        # 3. Numerical stability: subtract row-wise max (ignores -inf entries)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        # 4. Positive mask: exclude self-pairs
        mask = pos_mask.to(dtype=feats.dtype, device=device).clone()
        mask.fill_diagonal_(0.0)

        # 5. Denominator: sum exp over all non-self entries
        #    Diagonal is -inf → exp(-inf) = 0, so no masking needed here.
        exp_logits = torch.exp(logits)                            # [N, N]
        log_denom = torch.log(exp_logits.sum(dim=1, keepdim=True) + self.eps)

        # 6. log p(b | a) for all pairs
        log_prob = logits - log_denom                             # [N, N]

        # 7. For each anchor average log_prob over its positives.
        #    Use torch.where instead of mask * log_prob to avoid
        #    0 * (-inf) = nan (IEEE 754): non-positive positions get 0.
        pos_weight_sum = mask.sum(dim=1)                          # [N]
        valid = pos_weight_sum > 0                                # [N] bool

        if not valid.any():
            return torch.tensor(0.0, device=device, requires_grad=True)

        zero = torch.zeros_like(log_prob)
        masked_log_prob = torch.where(mask.bool(), log_prob, zero) # [N, N]
        mean_log_prob_pos = masked_log_prob.sum(dim=1)             # [N]
        mean_log_prob_pos[valid] = (
            mean_log_prob_pos[valid] / pos_weight_sum[valid]
        )
        mean_log_prob_pos[~valid] = 0.0

        # 8. Scale and average over valid anchors only
        loss_per_anchor = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        return loss_per_anchor[valid].mean()
