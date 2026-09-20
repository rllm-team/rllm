from __future__ import annotations

import copy

import pandas as pd
import torch

from rllm.data import GraphData, HeteroGraphData, TableData
from rllm.types import ColType


def test_table_data_tensorization_preserves_type_order_and_target_alignment():
    frame = pd.DataFrame(
        {
            "num_a": [1.0, 2.5, 4.0],
            "cat": [0, 1, 0],
            "num_b": [10.0, 20.0, 30.0],
            "target": [2, 0, 1],
        }
    )
    table = TableData(
        frame,
        {
            "num_a": ColType.NUMERICAL,
            "cat": ColType.CATEGORICAL,
            "num_b": ColType.NUMERICAL,
            "target": ColType.CATEGORICAL,
        },
        target_col="target",
    )

    assert table.feat_dict[ColType.NUMERICAL].dtype == torch.float32
    assert torch.equal(
        table.feat_dict[ColType.NUMERICAL],
        torch.tensor([[1.0, 10.0], [2.5, 20.0], [4.0, 30.0]]),
    )
    assert torch.equal(
        table.feat_dict[ColType.CATEGORICAL], torch.tensor([[0], [1], [0]])
    )
    assert torch.equal(table.y, torch.tensor([2.0, 0.0, 1.0]))


def test_table_data_tensor_slice_aligns_features_and_labels_without_copying_metadata():
    frame = pd.DataFrame({"feature": [5.0, 7.0, 11.0], "target": [0, 1, 0]})
    table = TableData(
        frame,
        {"feature": ColType.NUMERICAL, "target": ColType.CATEGORICAL},
        target_col="target",
    )

    sliced = table[torch.tensor([2, 0])]

    assert len(sliced) == 2
    assert torch.equal(sliced.feat_dict[ColType.NUMERICAL], torch.tensor([[11.0], [5.0]]))
    assert torch.equal(sliced.y, torch.tensor([0.0, 0.0]))
    assert sliced.df is table.df
    assert sliced.metadata is table.metadata


def test_graph_shallow_copy_and_clone_have_distinct_mutation_boundaries():
    graph = GraphData(x=torch.tensor([[1.0], [2.0]]), label="original")

    shallow = copy.copy(graph)
    shallow.label = "copy"
    cloned = graph.clone()
    cloned.x[0, 0] = 99.0

    assert graph.label == "original"
    assert shallow.x is graph.x
    assert graph.x[0, 0].item() == 1.0
    assert cloned.x is not graph.x


def test_heterogeneous_csc_uses_destination_cardinality_and_sorted_rows():
    graph = HeteroGraphData()
    graph["src"].num_nodes = 3
    graph["dst"].num_nodes = 5
    edge_type = ("src", "links", "dst")
    graph[edge_type].edge_index = torch.tensor(
        [[2, 0, 1], [4, 0, 4]], dtype=torch.long
    )

    colptr, row, permutation = graph.to_csc_dict()

    assert torch.equal(colptr[edge_type], torch.tensor([0, 1, 1, 1, 1, 3]))
    assert torch.equal(row[edge_type], torch.tensor([0, 2, 1]))
    assert torch.equal(permutation[edge_type], torch.tensor([1, 0, 2]))
