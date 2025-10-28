from __future__ import annotations

import torch
from typing import Literal

from rllm.nn.loss.base_loss import BaseContrastiveLoss


class SelfSupervisedVPCL(BaseContrastiveLoss):
    r"""
    Self-Supervised Vertical-Partition Contrastive Loss (Self-VPCL).

    This loss was proposed in
    *"TransTab: Learning Transferable Tabular Transformers Across Tables"*
    (<https://arxiv.org/abs/2205.09328>),
    as a self-supervised contrastive learning objective designed for tabular data.
    It extends InfoNCE-style contrastive learning to vertically partitioned tables,
    where each row is divided into multiple column subsets (partitions) treated as
    distinct "views" of the same sample.

    Positive pairs are formed between different partition embeddings derived from
    the **same row**, while negative pairs are formed across different rows.
    This formulation encourages consistency among embeddings of the same record
    under different column subsets, without relying on labels.

    Mathematical Formulation:

    The self-supervised VPCL loss is defined as:

    \[
    \ell(X) =
    - \sum_{i=1}^{B} \sum_{k=1}^{K} \sum_{k' \neq k}^{K}
    \log
    \frac{
        \exp\big(\psi(\mathbf{v}_i^{k}, \mathbf{v}_i^{k'})\big)
    }{
        \sum_{j=1}^{B}\sum_{k^{\dagger}=1}^{K}
        \exp\big(\psi(\mathbf{v}_i^{k}, \mathbf{v}_j^{k^{\dagger}})\big)
    } .
    \]

    where:
    - \( B \): batch size
    - \( K \): number of column partitions per sample
    - \( \mathbf{v}_i^{k} \): embedding of the \(k\)-th partition of the \(i\)-th row
    - \( \psi(\cdot, \cdot) \): similarity function (e.g., cosine or dot-product similarity)

    This objective maximizes the agreement between partition embeddings of the same
    record, while distinguishing them from embeddings belonging to other rows.

    Input Shapes
    - features: torch.Tensor of shape [B, K, D]
        - \( B \): batch size
        - \( K \): number of partitions per row
        - \( D \): projection dimension

    Arguments
    - temperature (float): Temperature \( \tau \) scaling the logits.
    - base_temperature (float): Reference temperature \( \tau_0 \) used for final scaling \( \tau / \tau_0 \).
    - similarity (str): Similarity metric; `"dot"` for raw dot product, `"cosine"` for L2-normalized cosine similarity.
    - eps (float): Numerical stability constant.
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
