from __future__ import annotations

import torch
from typing import Literal

from rllm.nn.loss.base_loss import BaseContrastiveLoss


class VerticalPartitionSupervisedLoss(BaseContrastiveLoss):
    r"""
    Supervised vertical partition contrastive loss(Supervised VPCL).

    This class implements the supervised variant of vertical-partition contrastive
    learning (VPCL). Each table row is split into multiple vertical partitions
    (i.e., column subsets), each partition is encoded into an embedding, and
    embeddings are trained to align across samples with the same class label.

    Positive pairs are formed between any two partition embeddings whose
    source rows share the same class label. Negative pairs are formed between
    partition embeddings from rows with different labels. This is analogous
    to supervised contrastive learning / supervised InfoNCE, but positives
    and negatives are defined at the (row, partition) level rather than just
    the row level.

    Expected input shapes:
        features (torch.Tensor): [B, K, D]
            B = batch size
            K = number of partitions (column subsets) per row
            D = projection dimension
        labels (torch.Tensor): [B]
            Integer class label for each row in the batch

    Args:
        temperature (float): Temperature τ used to scale pairwise logits.
        base_temperature (float): Normalization temperature τ₀ used in
            the final scaling factor (τ / τ₀).
        similarity (str): Similarity metric, "dot" for raw dot product or
            "cosine" for cosine similarity (L2-normalized dot product).
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

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute supervised vertical-partition contrastive loss.

        Args:
            features (torch.Tensor): Tensor of shape [B, K, D], where
                B is batch size, K is number of vertical partitions per row,
                and D is the projection dimension.
            labels (torch.Tensor): Tensor of shape [B], containing the
                integer class label for each row in the batch.

        Returns:
            torch.Tensor: Scalar loss.
        """
        device = features.device
        batch_size, num_partitions, _ = features.shape

        # Flatten [B, K, D] -> [B*K, D] so that each partition embedding
        # becomes an individual contrastive instance.
        feats = torch.cat(torch.unbind(features, dim=1), dim=0)  # [B*K, D]

        # Broadcast each row's label to all of its partitions:
        # labels_expanded: [B*K]
        labels_expanded = labels.to(device).long().repeat_interleave(num_partitions)

        # Build the positive-pair mask:
        # pos_mask[i, j] = 1 if feats[i] and feats[j] come from rows
        # with the same class label; else 0.
        pos_mask = (labels_expanded.unsqueeze(0) == labels_expanded.unsqueeze(1)).float()  # [B*K, B*K]

        # BaseContrastiveLoss handles:
        # - pairwise logits
        # - temperature scaling
        # - self-pair masking
        # - log-softmax and averaging over positives
        loss = super().forward(feats, pos_mask)
        return loss
