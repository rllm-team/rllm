from __future__ import annotations

import math

import torch

from rllm.nn.loss import ContrastiveLoss


def test_contrastive_loss_matches_independent_two_anchor_calculation():
    features = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], requires_grad=True)
    positives = torch.tensor([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=torch.float32)
    loss_fn = ContrastiveLoss(temperature=1.0, base_temperature=1.0, similarity="dot")

    loss = loss_fn(features, positives)

    expected = math.log(1.0 + math.e)
    assert torch.allclose(loss, torch.tensor(expected), atol=1e-6)
    loss.backward()
    assert features.grad is not None and torch.isfinite(features.grad).all()


def test_contrastive_loss_without_positives_returns_differentiable_zero():
    features = torch.randn(3, 2, requires_grad=True)
    loss = ContrastiveLoss()(features, torch.zeros(3, 3))

    assert loss.item() == 0.0
    assert loss.requires_grad
