from __future__ import annotations

import math

import torch

from rllm.transforms.graph_transforms import GCNNorm, NormalizeFeatures
from rllm.transforms.graph_transforms.functional import (
    add_remaining_self_loops,
    remove_self_loops,
)


def test_self_loop_transforms_replace_existing_diagonal_values_without_mutating_input():
    adjacency = torch.tensor([[3.0, 2.0], [4.0, 0.0]])

    with_loops = add_remaining_self_loops(adjacency, fill_value=7.0)
    without_loops = remove_self_loops(adjacency)

    assert torch.equal(with_loops, torch.tensor([[7.0, 2.0], [4.0, 7.0]]))
    assert torch.equal(without_loops, torch.tensor([[0.0, 2.0], [4.0, 0.0]]))
    assert torch.equal(adjacency, torch.tensor([[3.0, 2.0], [4.0, 0.0]]))


def test_gcn_norm_matches_hand_computed_symmetric_normalization():
    indices = torch.tensor([[0, 1], [1, 0]])
    adjacency = torch.sparse_coo_tensor(indices, torch.ones(2), (2, 2))

    normalized = GCNNorm()(adjacency).to_dense()

    assert torch.allclose(normalized, torch.full((2, 2), 0.5), atol=1e-7)


def test_feature_normalization_has_unit_rows_and_preserves_zero_row():
    features = torch.tensor([[3.0, 4.0], [0.0, 0.0], [1.0, 1.0]])

    normalized = NormalizeFeatures(norm="l2")(features.clone())

    expected = torch.tensor(
        [[0.6, 0.8], [0.0, 0.0], [1 / math.sqrt(2), 1 / math.sqrt(2)]]
    )
    assert torch.allclose(normalized, expected, atol=1e-7)
