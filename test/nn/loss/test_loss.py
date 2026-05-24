import pytest
import torch
import torch.nn.functional as F

from rllm.nn.loss import BaseLoss, ContrastiveLoss, SelfSupervisedVPCL, SupervisedVPCL


class DummyLoss(BaseLoss):
    def forward(self, x):
        return x.mean()


def manual_contrastive_loss(feats, pos_mask, temperature=1.0, base_temperature=1.0, similarity="dot"):
    if similarity == "cosine":
        feats = F.normalize(feats, dim=1)
    logits = feats @ feats.T / temperature
    eye = torch.eye(feats.size(0), dtype=torch.bool, device=feats.device)
    logits = logits.masked_fill(eye, float("-inf"))
    logits = logits - torch.max(logits, dim=1, keepdim=True).values.detach()
    mask = pos_mask.to(feats.dtype).clone()
    mask.fill_diagonal_(0.0)
    valid = mask.sum(dim=1) > 0
    log_prob = logits - torch.log(torch.exp(logits).sum(dim=1, keepdim=True) + 1e-12)
    per_anchor = torch.where(mask.bool(), log_prob, torch.zeros_like(log_prob)).sum(dim=1)
    per_anchor[valid] = per_anchor[valid] / mask.sum(dim=1)[valid]
    return (-(temperature / base_temperature) * per_anchor[valid]).mean()


def test_base_loss_is_abstract_and_subclass_returns_scalar():
    with pytest.raises(TypeError):
        BaseLoss()

    loss = DummyLoss()(torch.tensor([1.0, 2.0, 3.0]))

    assert loss.ndim == 0
    assert loss.item() == 2.0


def test_contrastive_loss_matches_manual_dot_formula_and_backpropagates():
    feats = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        requires_grad=True,
    )
    pos_mask = torch.tensor(
        [
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 0],
        ],
        dtype=torch.float32,
    )
    loss_fn = ContrastiveLoss(temperature=1.0, base_temperature=1.0, similarity="dot")

    loss = loss_fn(feats, pos_mask)
    expected = manual_contrastive_loss(feats, pos_mask)
    loss.backward()

    assert torch.allclose(loss, expected)
    assert feats.grad is not None
    assert torch.isfinite(feats.grad).all()


def test_contrastive_loss_cosine_and_no_positive_paths():
    feats = torch.randn(4, 3, requires_grad=True)
    pos_mask = torch.tensor(
        [
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
        ],
        dtype=torch.bool,
    )
    loss_fn = ContrastiveLoss(temperature=0.5, base_temperature=1.0, similarity="cosine")

    loss = loss_fn(feats, pos_mask)
    expected = manual_contrastive_loss(feats, pos_mask, temperature=0.5, base_temperature=1.0, similarity="cosine")

    assert torch.allclose(loss, expected)

    no_pos = loss_fn(feats, torch.zeros(4, 4))
    no_pos.backward()

    assert no_pos.item() == 0.0
    assert no_pos.requires_grad


def test_self_supervised_vpcl_equivalent_to_row_positive_mask():
    features = torch.randn(3, 2, 4, requires_grad=True)
    loss_fn = SelfSupervisedVPCL(temperature=1.0, base_temperature=1.0)

    loss = loss_fn(features)

    row_ids = torch.arange(3).repeat_interleave(2)
    pos_mask = (row_ids[:, None] == row_ids[None, :]).float()
    expected = ContrastiveLoss(temperature=1.0, base_temperature=1.0)(features.view(6, 4), pos_mask)
    loss.backward()

    assert torch.allclose(loss, expected)
    assert features.grad is not None


def test_supervised_vpcl_equivalent_to_label_positive_mask_and_imports():
    features = torch.randn(4, 2, 3)
    labels = torch.tensor([0, 1, 0, 2])
    loss_fn = SupervisedVPCL(temperature=1.0, base_temperature=1.0)

    loss = loss_fn(features, labels)

    expanded = labels.repeat_interleave(2)
    pos_mask = (expanded[:, None] == expanded[None, :]).float()
    expected = ContrastiveLoss(temperature=1.0, base_temperature=1.0)(features.view(8, 3), pos_mask)

    assert torch.allclose(loss, expected)
    assert isinstance(loss_fn, ContrastiveLoss)
