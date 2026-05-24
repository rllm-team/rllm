from collections import namedtuple

import numpy as np
import pytest
import torch

from rllm.data.storage import BaseStorage, EdgeStorage, NodeStorage, recursive_apply
from rllm.data.view import ItemsView, KeysView, ValuesView


def test_mapping_views_filter_iteration_len_and_repr():
    mapping = {"a": 1, "b": 2, "c": 3}

    keys = KeysView(mapping, "b", "missing", "a")
    values = ValuesView(mapping, "b", "a")
    items = ItemsView(mapping, "c", "a")

    assert list(keys) == ["b", "a"]
    assert list(values) == [2, 1]
    assert list(items) == [("c", 3), ("a", 1)]
    assert len(keys) == 2
    assert repr(items) == "ItemsView({'c': 3, 'a': 1})"


def test_base_storage_attribute_mapping_apply_copy_and_device_helpers():
    storage = BaseStorage({"x": torch.tensor([1.0, 2.0])}, y=[torch.tensor([3.0])])

    assert len(storage) == 2
    assert "x" in storage
    assert storage.x.tolist() == [1.0, 2.0]
    assert list(storage.keys("y", "missing", "x")) == ["y", "x"]
    assert storage.get("missing", "fallback") == "fallback"

    storage.apply(lambda x: x + 1 if torch.is_tensor(x) else x, "x")
    assert storage.x.tolist() == [2.0, 3.0]

    copied = storage.__copy__()
    copied.x = torch.tensor([9.0])
    assert storage.x.tolist() == [2.0, 3.0]

    assert storage.to_dict()["x"].tolist() == [2.0, 3.0]
    del storage.y
    assert "y" not in storage


def test_node_storage_num_nodes_and_node_attr_detection():
    storage = NodeStorage(x=torch.ones(3, 2), labels=np.array([0, 1, 0]), name=["a", "b", "c"])

    assert storage.num_nodes == 3
    assert storage.is_node_attr("x")
    assert storage.is_node_attr("labels")
    assert storage.is_node_attr("name")

    storage.bad = torch.ones(2, 1)
    assert not storage.is_node_attr("bad")
    assert NodeStorage().num_nodes == -1


def test_edge_storage_num_edges_csc_and_edge_attr_detection():
    edge_index = torch.tensor([[1, 0, 1], [2, 1, 0]])
    storage = EdgeStorage(edge_index=edge_index, weight=torch.tensor([0.1, 0.2, 0.3]))
    storage._key = ("user", "rates", "item")

    colptr, row, perm = storage.to_csc(num_nodes=3)

    assert storage.num_edges == 3
    assert storage.is_bipartite()
    assert storage.is_edge_attr("edge_index")
    assert storage.is_edge_attr("weight")
    assert colptr.tolist() == [0, 1, 2, 3]
    assert row.tolist() == [1, 0, 1]
    assert perm.tolist() == [2, 1, 0]

    with pytest.raises(AssertionError):
        storage.to_csc(edge_time="missing")


def test_edge_storage_sparse_adj_and_missing_edge_error():
    adj = torch.sparse_coo_tensor(
        torch.tensor([[0, 1], [1, 0]]),
        torch.ones(2),
        (2, 2),
    ).coalesce()
    storage = EdgeStorage(adj=adj)
    empty = EdgeStorage()

    assert storage.num_edges == 2
    colptr, row, perm = storage.to_csc(num_nodes=2)
    assert colptr.tolist() == [0, 1, 2]
    assert row.tolist() == [1, 0]
    assert perm is None

    with pytest.raises(ValueError):
        empty.to_csc()


def test_edge_storage_list_edge_attr_is_detected():
    storage = EdgeStorage(edge_index=torch.tensor([[0, 1], [1, 0]]), names=["a", "b"])

    assert storage.is_edge_attr("names")


def test_recursive_apply_nested_sequences_mappings_and_namedtuples():
    Pair = namedtuple("Pair", ["left", "right"])
    data = {
        "tensor": torch.tensor([1]),
        "list": [torch.tensor([2]), "keep"],
        "tuple": (torch.tensor([3]),),
        "named": Pair(torch.tensor([4]), "x"),
    }

    out = recursive_apply(data, lambda x: x + 10 if torch.is_tensor(x) else x)

    assert out["tensor"].tolist() == [11]
    assert out["list"][0].tolist() == [12]
    assert out["list"][1] == "keep"
    assert out["tuple"][0].tolist() == [13]
    assert isinstance(out["named"], Pair)
    assert out["named"].left.tolist() == [14]
