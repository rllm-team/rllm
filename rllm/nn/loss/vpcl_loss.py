from __future__ import annotations

import torch
from typing import Literal

from rllm.nn.loss.contrastive_loss import ContrastiveLoss


class SelfSupervisedVPCL(ContrastiveLoss):
    r"""
    The self-supervised vertical-partition contrastive loss (Self-VPCL)
    implementation, based on the
    `"TransTab: Learning Transferable Tabular Transformers Across Tables"
    <https://arxiv.org/abs/2205.09328>`__ paper.

    In this setting, each table row is split into multiple column partitions that
    act as different views of the same sample. Positive pairs are formed across
    partitions from the same row, while negatives are formed across rows.

    .. math::
        \ell(X) =
        - \sum_{i=1}^{B} \sum_{k=1}^{K} \sum_{k' \neq k}^{K}
        \log
        \frac{
            \exp\big(\psi(\mathbf{v}_i^{k}, \mathbf{v}_i^{k'})\big)
        }{
            \sum_{j=1}^{B}\sum_{k^{\dagger}=1}^{K}
            \exp\big(\psi(\mathbf{v}_i^{k}, \mathbf{v}_j^{k^{\dagger}})\big)
        } .

    where :math:`B` is batch size, :math:`K` is partition count per sample,
    :math:`\mathbf{v}_i^k` is the partition embedding, and :math:`\psi` is
    the configured similarity function.

    Args:
        temperature (float): Temperature :math:`\tau` scaling logits.
        base_temperature (float): Reference temperature :math:`\tau_0` used
            in the final scaling factor :math:`\tau / \tau_0`.
        similarity (str): Similarity metric, either ``"dot"`` or ``"cosine"``.
        eps (float): Numerical stability constant.

    Shapes:
        - **input:** partition embeddings :math:`(B, K, D)`
        - **output:** scalar loss :math:`()`

    Example:
        >>> import torch
        >>> loss_fn = SelfSupervisedVPCL()
        >>> feats = torch.randn(4, 2, 8)
        >>> loss_fn(feats).ndim
        0
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
        r"""Compute self-supervised vertical-partition contrastive loss.

        Args:
            features (torch.Tensor): Partition embeddings with shape
                :math:`(B, K, D)`.

        Returns:
            torch.Tensor: Scalar loss tensor with shape :math:`()`.

        Example:
            >>> import torch
            >>> loss_fn = SelfSupervisedVPCL()
            >>> feats = torch.randn(4, 2, 8)
            >>> out = loss_fn(feats)
            >>> out.shape
            torch.Size([])
        """
        device = features.device
        batch_size, num_partitions, _ = features.shape

        # Flatten [B, K, D] -> [B*K, D] via view (zero-copy, no extra allocation).
        feats = features.view(batch_size * num_partitions, -1)  # [B*K, D]

        # Assign each partition embedding an integer row id:
        # row_ids: [B*K], e.g. [0,0,...,0,1,1,...,1,...]
        row_ids = torch.arange(batch_size, device=device).repeat_interleave(
            num_partitions
        )

        # Build the positive-pair mask:
        # pos_mask[i, j] = 1 if feats[i] and feats[j] originate from the same row.
        pos_mask = (row_ids.unsqueeze(0) == row_ids.unsqueeze(1)).float()  # [B*K, B*K]

        # BaseContrastiveLoss handles
        loss = super().forward(feats, pos_mask)
        return loss


class SupervisedVPCL(ContrastiveLoss):
    r"""
    The supervised vertical-partition contrastive loss (Supervised-VPCL)
    implementation, based on the
    `"TransTab: Learning Transferable Tabular Transformers Across Tables"
    <https://arxiv.org/abs/2205.09328>`__ paper.

    It extends supervised contrastive learning to vertically partitioned tabular
    data. Positive pairs are built from partitions whose source rows share the
    same class label, while rows from different labels serve as negatives.

    .. math::
        \ell(X, y) =
        - \sum_{i=1}^{B} \sum_{j=1}^{B} \sum_{k=1}^{K} \sum_{k'=1}^{K}
        \mathbf{1}\{y_j = y_i\}
        \log
        \frac{
            \exp\big(\psi(\mathbf{v}_i^{k}, \mathbf{v}_j^{k'})\big)
        }{
            \sum_{j^{\dagger}=1}^{B}\sum_{k^{\dagger}=1}^{K}
            \mathbf{1}\{y_{j^{\dagger}} \neq y_i\}
            \exp\big(\psi(\mathbf{v}_i^{k}, \mathbf{v}_{j^{\dagger}}^{k^{\dagger}})\big)
        } .

    where :math:`B` is batch size, :math:`K` is partition count per sample,
    :math:`\mathbf{v}_i^k` is the partition embedding, and :math:`y_i` is
    the class label for sample :math:`i`.

    Args:
        temperature (float): Temperature :math:`\tau` scaling logits.
        base_temperature (float): Reference temperature :math:`\tau_0` used
            in the final scaling factor :math:`\tau / \tau_0`.
        similarity (str): Similarity metric, either ``"dot"`` or ``"cosine"``.
        eps (float): Numerical stability constant.

    Shapes:
        - **input:**
            partition embeddings :math:`(B, K, D)`
            labels :math:`(B,)`
        - **output:** scalar loss :math:`()`

    Example:
        >>> import torch
        >>> loss_fn = SupervisedVPCL()
        >>> feats = torch.randn(4, 2, 8)
        >>> labels = torch.tensor([0, 1, 0, 1])
        >>> loss_fn(feats, labels).shape
        torch.Size([])
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
        r"""Compute supervised vertical-partition contrastive loss.

        Args:
            features (torch.Tensor): Partition embeddings with shape
                :math:`(B, K, D)`.
            labels (torch.Tensor): Class labels with shape :math:`(B,)`.

        Returns:
            torch.Tensor: Scalar loss tensor with shape :math:`()`.

        Example:
            >>> import torch
            >>> loss_fn = SupervisedVPCL()
            >>> feats = torch.randn(4, 2, 8)
            >>> labels = torch.tensor([0, 1, 0, 1])
            >>> out = loss_fn(feats, labels)
            >>> out.shape
            torch.Size([])
        """
        device = features.device
        batch_size, num_partitions, _ = features.shape

        # Flatten [B, K, D] -> [B*K, D] via view (zero-copy, no extra allocation).
        feats = features.view(batch_size * num_partitions, -1)  # [B*K, D]

        # Broadcast each row's label to all of its partitions:
        # labels_expanded: [B*K]
        labels_expanded = labels.to(device).long().repeat_interleave(num_partitions)

        # Build the positive-pair mask:
        # pos_mask[i, j] = 1 if feats[i] and feats[j] come from rows
        # with the same class label; else 0.
        pos_mask = (
            labels_expanded.unsqueeze(0) == labels_expanded.unsqueeze(1)
        ).float()  # [B*K, B*K]

        # BaseContrastiveLoss handles
        loss = super().forward(feats, pos_mask)
        return loss
