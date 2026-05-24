import copy

import pytest
import torch

from rllm.data import GraphData, HeteroGraphData


def sparse_adj(indices, values, size):
    return torch.sparse_coo_tensor(
        torch.tensor(indices, dtype=torch.long),
        torch.tensor(values, dtype=torch.float32),
        size,
    ).coalesce()


def test_graph_data_mapping_properties_save_load_and_apply(tmp_path):
    graph = GraphData(
        x=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        y=torch.tensor([0, 1]),
        adj=sparse_adj([[0, 1], [1, 0]], [1.0, 1.0], (2, 2)),
        train_mask=torch.tensor([True, False]),
    )

    assert len(graph) == 2
    assert graph.num_nodes == 2
    assert graph.num_classes == 2
    assert "x" in graph
    assert set(graph.keys()) == {"x", "y", "adj", "train_mask", "num_classes"}
    assert graph.to_dict()["train_mask"].tolist() == [True, False]

    graph["extra"] = torch.tensor([5])
    assert graph.extra.tolist() == [5]
    del graph["extra"]
    with pytest.raises(KeyError):
        _ = graph["extra"]

    cloned = graph.clone("x")
    cloned.x[0, 0] = 99.0
    assert graph.x[0, 0].item() == 1.0

    graph.apply(lambda x: x + 1 if torch.is_tensor(x) and x.dtype.is_floating_point else x, "x")
    assert graph.x.tolist() == [[2.0, 3.0], [4.0, 5.0]]

    path = tmp_path / "graph.pt"
    graph.save(path)
    loaded = GraphData.load(path)
    assert loaded.x.tolist() == graph.x.tolist()
    assert loaded.y.tolist() == [0, 1]


def test_graph_data_num_nodes_fallbacks_and_errors():
    assert GraphData(y=torch.tensor([1, 2, 3])).num_nodes == 3
    assert GraphData(adj=sparse_adj([[0], [1]], [1.0], (4, 4))).num_nodes == 4

    with pytest.raises(AttributeError):
        _ = GraphData().num_nodes


def test_graph_data_to_hetero_splits_nodes_edges_and_attrs():
    graph = GraphData(
        x=torch.tensor([[1.0], [2.0], [3.0], [4.0]]),
        adj=sparse_adj([[0, 2, 1, 3], [1, 3, 0, 2]], [1.0, 2.0, 3.0, 4.0], (4, 4)),
        note="kept",
    )
    node_type = torch.tensor([0, 0, 1, 1])
    edge_type = torch.tensor([0, 1, 0, 1])
    edge_names = [("paper", "cites", "paper"), ("author", "knows", "author")]

    hetero = graph.to_hetero(
        node_type=node_type,
        edge_type=edge_type,
        node_type_names=["paper", "author"],
        edge_type_names=edge_names,
    )

    assert hetero["paper"].x.tolist() == [[1.0], [2.0]]
    assert hetero["author"].x.tolist() == [[3.0], [4.0]]
    assert hetero[edge_names[0]].adj.shape == (2, 2)
    assert hetero.note == "kept"
    assert hetero.metadata() == (["paper", "author"], edge_names)


def test_graph_data_to_hetero_rejects_edge_type_spanning_node_types():
    graph = GraphData(
        x=torch.ones(4, 1),
        adj=sparse_adj([[0, 2], [1, 3]], [1.0, 1.0], (4, 4)),
    )
    node_type = torch.tensor([0, 0, 1, 1])
    edge_type = torch.tensor([0, 0])

    with pytest.raises(ValueError):
        graph.to_hetero(node_type=node_type, edge_type=edge_type, node_type_names=["a", "b"])


def test_heterographdata_construction_access_collect_validate_and_copy():
    hetero = HeteroGraphData(
        {
            "user": {"x": torch.tensor([[1.0], [2.0]])},
            "item": {"x": torch.tensor([[3.0], [4.0], [5.0]])},
            ("user", "rates", "item"): {
                "edge_index": torch.tensor([[0, 1], [1, 2]]),
                "weight": torch.tensor([4.0, 5.0]),
            },
        },
        graph_name="toy",
    )

    assert hetero.graph_name == "toy"
    assert hetero.num_nodes == 5
    assert hetero.node_types == ["user", "item"]
    assert hetero.edge_types == [("user", "rates", "item")]
    assert hetero.validate()
    assert hetero.x_dict()["user"].tolist() == [[1.0], [2.0]]
    assert hetero.collect_attr("x")["item"].tolist() == [[3.0], [4.0], [5.0]]

    hetero["user"].x = None
    assert "user" in hetero.collect_attr("x", exlude_None=False)
    assert "user" not in hetero.collect_attr("x", exlude_None=True)

    copied = copy.copy(hetero)
    assert copied is not hetero
    assert copied["item"].x is hetero["item"].x
    copied["item"].x = torch.tensor([[9.0]])
    assert copied["item"].x is not hetero["item"].x


def test_heterographdata_key_regulation_save_load_csc_and_set_value(tmp_path):
    hetero = HeteroGraphData()
    hetero["user"].x = torch.ones(2, 1)
    hetero["item"].x = torch.ones(3, 1)
    hetero["user__rates__item"].edge_index = torch.tensor([[1, 0, 1], [2, 1, 0]])
    hetero["user", "rates", "item"].edge_time = torch.tensor([30, 10, 20])

    colptr_d, row_d, perm_d = hetero.to_csc_dict(edge_time_d={("user", "rates", "item"): hetero["user", "rates", "item"].edge_time})
    edge_type = ("user", "rates", "item")

    assert colptr_d[edge_type].tolist() == [0, 1, 2]
    assert row_d[edge_type].tolist() == [1, 0, 1]
    assert perm_d[edge_type].tolist() == [2, 1, 0]

    hetero.set_value_dict("batch", {"user": torch.tensor([0, 0]), edge_type: torch.tensor([1, 1, 1])})
    assert hetero["user"].batch.tolist() == [0, 0]
    assert hetero[edge_type].batch.tolist() == [1, 1, 1]

    path = tmp_path / "hetero.pt"
    hetero.save(path)
    loaded = HeteroGraphData.load(path)
    assert loaded.node_types == ["user", "item"]
    assert loaded.edge_types == [edge_type]

    del loaded[edge_type]
    assert loaded.edge_types == []
    del loaded["user"]
    assert loaded.node_types == ["item"]


def test_heterographdata_validate_detects_invalid_edges():
    hetero = HeteroGraphData()
    hetero["src"].x = torch.ones(2, 1)
    hetero["dst"].x = torch.ones(2, 1)
    hetero["src", "bad", "dst"].edge_index = torch.tensor([[0, 2], [1, -1]])

    assert not hetero.validate()


def test_heterographdata_repr_and_adj_dict():
    hetero = HeteroGraphData()
    hetero["n"].x = torch.ones(2, 1)
    hetero["n", "self", "n"].adj = sparse_adj([[0], [1]], [1.0], (2, 2))

    assert hetero.adj_dict()[("n", "self", "n")].shape == (2, 2)
    assert "HeteroGraphData" in repr(hetero)

