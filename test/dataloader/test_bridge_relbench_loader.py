import types

import pandas as pd
import pytest
import torch

from rllm.data import GraphData, HeteroGraphData, TableData
from rllm.dataloader.bridge_loader import BRIDGELoader
from rllm.dataloader.relbench_loader import AttachTargetTransform, RelbenchLoader
from rllm.dataloader.sampler.data_type import HeteroSamplerOutput
from rllm.datasets import RelBenchTaskType
from rllm.types import ColType


def make_table(num_rows=4):
    return TableData(
        pd.DataFrame({"x": list(range(num_rows)), "target": [0, 1, 0, 1][:num_rows]}),
        {"x": ColType.NUMERICAL, "target": ColType.CATEGORICAL},
        target_col="target",
    )


def test_bridge_loader_splits_table_and_non_table_nodes():
    table = make_table(4)
    non_table = torch.tensor([[10.0], [11.0], [12.0]])
    graph = GraphData(
        edge_index=torch.tensor([[0, 1, 4, 5], [1, 4, 5, 6]]),
        num_nodes=7,
    )
    graph.device = torch.device("cpu")

    loader = BRIDGELoader(
        table=table,
        non_table=non_table,
        graph=graph,
        num_samples=[-1],
        train_mask=torch.tensor([6]),
        batch_size=1,
    )

    batch, n_id, adjs, table_data, non_table_data = next(iter(loader))

    assert batch == 1
    assert n_id[0].item() == 6
    assert len(table_data) == int((n_id < len(table)).sum().item())
    assert non_table_data.shape[0] == int((n_id >= len(table)).sum().item())


def test_bridge_loader_handles_missing_non_table_features():
    table = make_table(3)
    graph = GraphData(edge_index=torch.tensor([[0, 1], [1, 2]]), num_nodes=3)
    graph.device = torch.device("cpu")
    loader = BRIDGELoader(table, None, graph, num_samples=[-1], train_mask=torch.tensor([2]), batch_size=1)

    _, n_id, _, table_data, non_table_data = next(iter(loader))

    assert len(table_data) == len(n_id)
    assert non_table_data is None


def test_attach_target_transform_sets_labels_by_input_id():
    batch = HeteroGraphData()
    batch["user"].input_id = torch.tensor([2, 0])
    target = torch.tensor([10.0, 20.0, 30.0])

    out = AttachTargetTransform("user", target)(batch)

    assert out["user"].y.tolist() == [30.0, 10.0]


def test_relbench_loader_static_filter_node_and_edge_store():
    table = make_table(4)
    src = HeteroGraphData()
    src["user"].num_nodes = 4
    src["user"].time = torch.tensor([100, 200, 300, 400])
    src["user"].table = table
    src["item"].num_nodes = 2
    src["item"].time = torch.tensor([10, 20])
    src["user", "rates", "item"].edge_index = torch.tensor([[0, 1, 3], [1, 0, 1]])

    out = src.__copy__()
    RelbenchLoader.filter_node_store_(src["user"], out["user"], torch.tensor([3, 1]))
    RelbenchLoader.filter_edge_store_(
        src["user", "rates", "item"],
        out["user", "rates", "item"],
        torch.tensor([1, 0]),
        torch.tensor([0, 1]),
    )

    assert out["user"].num_nodes == 2
    assert out["user"].time.tolist() == [400, 200]
    assert len(out["user"].table) == 2
    assert out["user", "rates", "item"].edge_index.tolist() == [[1, 0], [0, 1]]


def test_relbench_loader_filter_edge_store_rejects_extra_edge_attrs():
    store = types.SimpleNamespace()
    source = {"edge_index": torch.tensor([[0], [1]]), "weight": torch.tensor([1.0])}

    class SourceStore(dict):
        def items(self):
            return super().items()

    with pytest.raises(NotImplementedError):
        RelbenchLoader.filter_edge_store_(SourceStore(source), store, torch.tensor([0]), torch.tensor([1]))


def test_relbench_loader_builds_sampler_input_and_filters_batch(tmp_path):
    class FakeTask:
        task_name = "task"
        entity_col = "user_id"
        time_col = "time"
        target_col = "target"
        task_type = RelBenchTaskType.BINARY_CLASSIFICATION
        entity_table = "user"

        def __init__(self):
            df = pd.DataFrame(
                {
                    "user_id": [0, 1],
                    "time": pd.to_datetime(["1970-01-01", "1970-01-02"]),
                    "target": [1.0, 0.0],
                }
            )
            self.task_data_dict = {
                "train": (df, None),
                "val": (df, None),
                "test": (df, None),
            }

    class FakeDataset:
        tasks = ["task"]

        def __init__(self):
            self.task_dict = {"task": FakeTask()}
            self.hdata = HeteroGraphData()
            self.hdata["user"].num_nodes = 2
            self.hdata["user"].time = torch.tensor([0, 86400])
            self.hdata["user"].feat = torch.tensor([10, 20])
            self.hdata["item"].num_nodes = 1
            self.hdata["item"].time = torch.tensor([0])
            self.hdata["user", "buys", "item"].edge_index = torch.tensor([[0, 1], [0, 0]])

        def load_all(self):
            self.loaded = True

    loader = RelbenchLoader(
        dataset=FakeDataset(),
        task="task",
        split="train",
        batch_size=2,
        num_neighbors=[-1],
        use_pyg_lib=False,
    )

    sampled = HeteroSamplerOutput(
        node={"user": torch.tensor([0, 1]), "item": torch.tensor([0])},
        row={("user", "buys", "item"): torch.tensor([0, 1])},
        col={("user", "buys", "item"): torch.tensor([0, 0])},
        batch={"user": torch.tensor([0, 1]), "item": torch.tensor([0])},
        num_sampled_nodes={"user": [2], "item": [1]},
        num_sampled_edges={("user", "buys", "item"): [2]},
        metadata=(torch.tensor([0, 1]), torch.tensor([0, 86400])),
    )
    batch = loader.filter_fn(sampled)

    assert loader.sampler_input.node.tolist() == [0, 1]
    assert batch["user"].y.tolist() == [1.0, 0.0]
    assert batch["user"].n_id.tolist() == [0, 1]
    assert batch["user"].batch_size == 2
    assert batch.edge_index_dict[("user", "buys", "item")].tolist() == [[0, 1], [0, 0]]
    assert repr(loader) == "RelbenchLoader()"

    loaders = RelbenchLoader.get_loaders(FakeDataset(), "task", batch_size=1, num_neighbors=[-1])
    assert [loader.split for loader in loaders] == ["train", "val", "test"]
