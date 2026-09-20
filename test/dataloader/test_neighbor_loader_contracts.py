from __future__ import annotations

import torch

from rllm.data import GraphData
from rllm.dataloader import NeighborLoader


def _small_graph() -> GraphData:
    # Directed edges: 0->2, 1->2, 2->3. Node 4 is isolated.
    return GraphData(
        x=torch.arange(5, dtype=torch.float32).view(-1, 1),
        edge_index=torch.tensor([[0, 1, 2], [2, 2, 3]], dtype=torch.long),
    )


def test_neighbor_loader_returns_seed_first_global_ids_and_local_edges():
    loader = NeighborLoader(
        _small_graph(), num_neighbors=[-1], seeds=[2], batch_size=1, shuffle=False
    )

    batch_size, node_ids, adjacencies = next(iter(loader))
    adjacency = adjacencies[0].coalesce()

    assert batch_size == 1
    assert torch.equal(node_ids, torch.tensor([2, 0, 1]))
    assert torch.equal(adjacency.indices(), torch.tensor([[1, 2], [0, 0]]))
    assert torch.equal(adjacency.values(), torch.ones(2))


def test_neighbor_loader_accepts_boolean_seed_mask_and_isolated_node():
    seed_mask = torch.tensor([False, False, False, False, True])
    loader = NeighborLoader(
        _small_graph(), num_neighbors=[-1], seeds=seed_mask, batch_size=1
    )

    batch_size, node_ids, adjacencies = next(iter(loader))

    assert batch_size == 1
    assert torch.equal(node_ids, torch.tensor([4]))
    assert adjacencies[0].numel() == 0
