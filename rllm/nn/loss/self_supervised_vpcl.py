from __future__ import annotations

import torch
from typing import Literal

from rllm.nn.loss.base_loss import BaseContrastiveLoss


class VerticalPartitionSelfSupervisedLoss(BaseContrastiveLoss):
    r"""
    Self-supervised vertical-partition contrastive loss(Self-supervised VPCL).

    This class implements the self-supervised variant of vertical-partition
    contrastive learning (VPCL). Each row in a table is split into multiple
    vertical partitions (i.e., column subsets), each partition is encoded into
    an embedding, and embeddings from the *same row* are encouraged to align
    even without labels.

    Positive pairs are formed between any two partition embeddings that come
    from the same original row. Negative pairs are formed between partition
    embeddings from different rows. This follows the InfoNCE / NT-Xent
    contrastive learning formulation, but treats different column subsets of
    the same row as different "views."

    Expected input shapes:
        features (torch.Tensor): [B, K, D]
            B = batch size
            K = number of partitions (column subsets) per row
            D = projection dimension

    Args:
        temperature (float): Temperature τ for scaling pairwise logits.
        base_temperature (float): Reference temperature τ₀ for final scaling
            (τ / τ₀).
        similarity (str): "dot" for raw dot product similarity or "cosine" for
            cosine similarity (L2-normalized dot product).
        eps (float): Numerical stability constant.
    """

    def __init__(
        self,
        temperature: float = 10.0,
        base_temperature: float = 10.0,
        similarity: Literal["dot", "cosine"] = "dot",
        eps: float = 1e-12,
    ) -> None:
        super().__init__(
            temperature=temperature,
            base_temperature=base_temperature,
            similarity=similarity,
            eps=eps,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute self-supervised vertical-partition contrastive loss.

        Args:
            features (torch.Tensor): Tensor of shape [B, K, D], where
                B is batch size, K is number of vertical partitions per row,
                and D is the projection dimension.

        Returns:
            torch.Tensor: Scalar loss.
        """
        device = features.device
        batch_size, num_partitions, _ = features.shape

        # Flatten [B, K, D] -> [B*K, D] so that each partition embedding
        # becomes an individual contrastive instance.
        feats = torch.cat(torch.unbind(features, dim=1), dim=0)  # [B*K, D]

        # Assign each partition embedding an integer row id:
        # row_ids: [B*K], e.g. [0,0,...,0,1,1,...,1,...]
        row_ids = torch.arange(batch_size, device=device).repeat_interleave(num_partitions)

        # Build the positive-pair mask:
        # pos_mask[i, j] = 1 if feats[i] and feats[j] originate from the same row.
        pos_mask = (row_ids.unsqueeze(0) == row_ids.unsqueeze(1)).float()  # [B*K, B*K]

        # BaseContrastiveLoss handles:
        # - pairwise logits
        # - temperature scaling
        # - self-pair masking
        # - log-softmax and averaging over positives
        loss = super().forward(feats, pos_mask)
        return loss
