import pytest
import torch

from rllm.data import GraphData
from rllm.dataloader.neighbor_loader import NeighborLoader


def make_graph():
    edge_index = torch.tensor(
        [
            [0, 0, 1, 2, 2, 3, 4, 4, 5],
            [1, 2, 3, 3, 4, 4, 5, 6, 6],
        ]
    )
    graph = GraphData(edge_index=edge_index, x=torch.arange(7).float().view(-1, 1), num_nodes=7)
    graph.device = torch.device("cpu")
    return graph


def test_neighbor_loader_defaults_to_cpu_when_graph_has_no_device():
    graph = GraphData(
        edge_index=torch.tensor([[0, 1], [1, 2]]),
        x=torch.arange(3).float().view(-1, 1),
        num_nodes=3,
    )

    loader = NeighborLoader(graph, num_neighbors=[-1], seeds=[2], batch_size=1)

    assert loader.device == torch.device("cpu")


def test_neighbor_loader_builds_csc_and_in_neighbors():
    loader = NeighborLoader(make_graph(), num_neighbors=[-1], seeds=[5, 4], batch_size=2)

    assert loader.get_in_neighbors(6).tolist() == [4, 5]
    assert loader.get_in_neighbors(0).tolist() == []
    assert loader.col_ptr.tolist() == [0, 0, 1, 2, 4, 6, 7, 9]


def test_neighbor_loader_collate_samples_multi_hop_and_transform():
    def double_values(adj):
        adj = adj.coalesce()
        return torch.sparse_coo_tensor(adj.indices(), adj.values() * 2, adj.shape)

    loader = NeighborLoader(
        make_graph(),
        num_neighbors=[-1, 1],
        seeds=torch.tensor([6]),
        transform=double_values,
        batch_size=1,
        shuffle=False,
    )

    batch_size, n_id, adjs = next(iter(loader))

    assert batch_size == 1
    assert n_id[0].item() == 6
    assert len(adjs) == 2
    assert all(adj.is_sparse for adj in adjs)
    assert adjs[0].coalesce().values().tolist() == [2.0, 2.0]
    assert set(n_id.tolist()).issuperset({4, 5, 6})


def test_neighbor_loader_accepts_boolean_seed_mask_and_empty_neighbors():
    mask = torch.tensor([False, False, False, False, False, False, True])
    loader = NeighborLoader(make_graph(), num_neighbors=[0], seeds=mask, batch_size=1)

    batch_size, n_id, adjs = next(iter(loader))

    assert batch_size == 1
    assert n_id.tolist() == [6]
    assert adjs[0].numel() == 0


def test_neighbor_loader_requires_edges_for_csc_build():
    graph = GraphData(x=torch.ones(2, 1), num_nodes=2)
    graph.device = torch.device("cpu")

    with pytest.raises(UnboundLocalError):
        NeighborLoader(graph, num_neighbors=[1], seeds=[0])
