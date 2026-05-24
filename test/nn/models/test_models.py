import os

import pytest
import torch

from rllm.data import HeteroGraphData
from rllm.nn.models import (
    BRIDGE,
    GraphEncoder,
    HeteroSAGE,
    RDL,
    RECT_L,
    RelGNN,
    RelGNNModel,
    TableEncoder,
    TableResNet,
    TransTab,
    TransTabClassifier,
    TransTabForCL,
)
from rllm.nn.models.transtab import LinearClassifier, TransTabCLSToken
from rllm.types import ColType, StatType


class FakeTable:
    def __init__(self, feat_dict, colname_token_ids=None, df=None, col_types=None):
        self.feat_dict = feat_dict
        self.colname_token_ids = colname_token_ids or {}
        self.df = df
        self.col_types = col_types or {}

    def __len__(self):
        first = next(iter(self.feat_dict.values()))
        if isinstance(first, tuple):
            return first[0].size(0)
        return first.size(0)

    def if_materialized(self):
        return True


class FakeTokenizer:
    vocab_size = 64
    pad_token_id = 0

    def __init__(self):
        self.model_max_length = None

    def __call__(
        self,
        texts,
        padding=True,
        truncation=True,
        add_special_tokens=False,
        return_tensors="pt",
    ):
        if isinstance(texts, str):
            texts = [texts]
        rows = []
        for text in texts:
            tokens = [((ord(ch) % 50) + 1) for ch in text.replace(" ", "")[:4]]
            rows.append(tokens or [0])
        width = max(len(row) for row in rows)
        input_ids = torch.zeros(len(rows), width, dtype=torch.long)
        attention_mask = torch.zeros(len(rows), width, dtype=torch.long)
        for i, row in enumerate(rows):
            input_ids[i, : len(row)] = torch.tensor(row)
            attention_mask[i, : len(row)] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def save_pretrained(self, path):
        os.makedirs(path, exist_ok=True)
        return (path,)


def metadata():
    return {
        ColType.NUMERICAL: [
            {StatType.MEAN: 0.0, StatType.STD: 1.0},
            {StatType.MEAN: 1.0, StatType.STD: 2.0},
        ],
        ColType.CATEGORICAL: [
            {StatType.COUNT: 4},
        ],
    }


def make_table(num_rows=3):
    return FakeTable(
        {
            ColType.NUMERICAL: torch.randn(num_rows, 2),
            ColType.CATEGORICAL: torch.tensor([[0], [1], [2]])[:num_rows],
        }
    )


def make_transtab_table(batch_size=3):
    colname_token_ids = {
        "num_a": (torch.tensor([1, 2]), torch.tensor([1, 1])),
        "num_b": (torch.tensor([3, 4]), torch.tensor([1, 1])),
    }
    return FakeTable(
        {
            ColType.TEXT: (
                torch.tensor([[1, 2, 0], [3, 4, 5], [6, 0, 0]])[:batch_size],
                torch.tensor([[1, 1, 0], [1, 1, 1], [1, 0, 0]])[:batch_size],
            ),
            ColType.NUMERICAL: torch.randn(batch_size, 2),
            ColType.BINARY: torch.tensor([[1.0], [0.0], [1.0]])[:batch_size],
        },
        colname_token_ids=colname_token_ids,
    )


def test_table_resnet_rect_and_heterosage_forward_shapes():
    table_model = TableResNet(hidden_dim=4, out_dim=2, num_layers=1, metadata=metadata(), dropout=0.0)
    table_out = table_model(make_table(3))

    assert table_out.shape == (3, 2)

    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])
    rect = RECT_L(in_dim=5, hidden_dim=3, dropout=0.0)
    x = torch.randn(4, 5)
    y = torch.tensor([0, 1, 0, 1])
    mask = torch.tensor([True, True, True, False])

    assert rect(x, edge_index).shape == (4, 5)
    assert rect.embed(x, edge_index).shape == (4, 3)
    assert rect.get_semantic_labels(x, y, mask).shape == (3, 5)
    assert str(rect) == "RECT_L(5, 3)"

    sage = HeteroSAGE(["user", "item"], [("user", "rates", "item")], hidden_dim=4, num_layers=1)
    out = sage(
        {"user": torch.randn(3, 4), "item": torch.randn(2, 4)},
        {("user", "rates", "item"): torch.tensor([[0, 1, 2], [0, 0, 1]])},
    )

    assert out["user"].shape == (3, 4)
    assert out["item"].shape == (2, 4)


def test_graph_encoder_forward_full_and_layerwise_adj():
    graph_encoder = GraphEncoder(in_dim=8, out_dim=2, dropout=0.0, num_layers=2, norm=False)
    adj = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]])

    assert graph_encoder(torch.randn(5, 8), adj).shape == (5, 2)
    assert graph_encoder(torch.randn(5, 8), [adj, adj]).shape == (5, 2)


def test_bridge_table_encoder_and_bridge_forward():
    table_encoder = TableEncoder(in_dim=1, out_dim=8, num_layers=1, metadata=metadata())
    graph_encoder = GraphEncoder(in_dim=8, out_dim=2, dropout=0.0, num_layers=2, norm=False)
    table = make_table(3)
    non_table = torch.randn(2, 8)
    adj = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]])
    model = BRIDGE(table_encoder, graph_encoder)

    out = model(table, non_table, adj)

    assert table_encoder(table).shape == (3, 8)
    assert out.shape == (3, 2)


def test_transtab_base_classifier_cls_token_and_update():
    table = make_transtab_table(3)
    common = dict(
        categorical_columns=["cat_text"],
        numerical_columns=["num_a", "num_b"],
        binary_columns=["bin_a"],
        hidden_dim=4,
        num_layer=1,
        num_attention_head=2,
        hidden_dropout_prob=0.0,
        ffn_dim=8,
        tokenizer=FakeTokenizer(),
    )
    model = TransTab(**common)
    cls = model(table)

    assert cls.shape == (3, 4)

    cls_token = TransTabCLSToken(hidden_dim=4)
    cls_out = cls_token(torch.randn(2, 3, 4), attention_mask=torch.ones(2, 3))
    assert cls_out["embedding"].shape == (2, 4, 4)
    assert cls_out["attention_mask"].shape == (2, 4)

    clf = LinearClassifier(num_class=3, hidden_dim=4)
    assert clf(torch.randn(2, 4)).shape == (2, 3)

    binary = TransTabClassifier(num_class=2, **common)
    logits, loss = binary(table, torch.tensor([0.0, 1.0, 0.0]))
    assert logits.shape == (3, 1)
    assert loss.ndim == 0

    multi = TransTabClassifier(num_class=3, **common)
    logits, loss = multi(table, torch.tensor([0, 1, 2]))
    assert logits.shape == (3, 3)
    assert loss.ndim == 0
    multi.update({"num_class": 2})
    assert multi.clf.fc.out_features == 1

    with pytest.raises(ValueError):
        model({ColType.NUMERICAL: torch.randn(3, 2)})


def test_transtab_for_cl_helpers_and_invalid_configuration():
    model = TransTabForCL(
        categorical_columns=["cat_text"],
        numerical_columns=["num_a", "num_b"],
        binary_columns=["bin_a"],
        hidden_dim=4,
        num_layer=1,
        num_attention_head=2,
        hidden_dropout_prob=0.0,
        ffn_dim=8,
        projection_dim=3,
        num_partition=2,
        overlap_ratio=0.5,
        tokenizer=FakeTokenizer(),
    )

    assert model._infer_col_types(["cat_text", "num_a", "bin_a", "unknown"]) == {
        "cat_text": ColType.TEXT,
        "num_a": ColType.NUMERICAL,
        "bin_a": ColType.BINARY,
        "unknown": ColType.TEXT,
    }

    import pandas as pd

    df = pd.DataFrame({"cat_text": ["a", "b"], "num_a": [1.0, 2.0], "num_b": [3.0, 4.0], "bin_a": [1, 0]})
    parts = model._build_positive_pairs(df, 2)
    assert len(parts) == 2
    assert all(len(part) == len(df) for part in parts)

    with pytest.raises(AssertionError):
        TransTabForCL(num_partition=0, tokenizer=FakeTokenizer())
    with pytest.raises(AssertionError):
        TransTabForCL(overlap_ratio=1.0, tokenizer=FakeTokenizer())


def make_hetero_data():
    data = HeteroGraphData()
    data["user"].num_nodes = 3
    data["item"].num_nodes = 2
    data["user"].time = torch.tensor([0.0, 1.0, 2.0])
    data["item"].time = torch.tensor([0.0, 1.0])
    data["user", "rates", "item"].edge_index = torch.tensor([[0, 1, 2], [0, 0, 1]])
    return data


def make_batch():
    batch = make_hetero_data()
    batch["user"].table = make_table(3)
    batch["item"].table = make_table(2)
    batch["item"].seed_time = torch.tensor([10.0, 20.0])
    batch.time_dict = {
        "user": torch.tensor([1.0, 2.0, 3.0]),
        "item": torch.tensor([4.0, 5.0]),
    }
    batch.batch_dict = {"user": torch.tensor([0, 0, 1]), "item": torch.tensor([0, 1])}
    batch.edge_index_dict = {("user", "rates", "item"): torch.tensor([[0, 1, 2], [0, 0, 1]])}
    return batch


def test_rdl_and_relgnn_models_forward_and_validation():
    data = make_hetero_data()
    stats = {"user": metadata(), "item": metadata()}
    batch = make_batch()

    rdl = RDL(
        data=data,
        col_stats_dict=stats,
        hidden_dim=4,
        out_dim=2,
        tnn_hidden_dim=4,
        tnn_num_layers=1,
        hgnn_num_layers=1,
        use_temporal_encoder=True,
    )
    rdl.eval()
    assert rdl(batch, "item").shape == (2, 2)

    relgnn = RelGNN(
        node_types=["user", "item"],
        atomic_routes_edge_types=[("dim-dim", "user", "rates", "item")],
        hidden_dim=4,
        num_layers=1,
        num_heads=1,
    )
    out = relgnn(
        {"user": torch.randn(3, 4), "item": torch.randn(2, 4)},
        {("user", "rates", "item"): torch.tensor([[0, 1, 2], [0, 0, 1]])},
    )
    assert out["item"].shape == (2, 4)

    rel_model = RelGNNModel(
        data=data,
        col_stats_dict=stats,
        atomic_routes_edge_types=[("dim-dim", "user", "rates", "item")],
        hidden_dim=4,
        out_dim=2,
        tnn_hidden_dim=4,
        tnn_num_layers=1,
        relgnn_num_layers=1,
        relgnn_num_heads=1,
        use_temporal_encoder=True,
    )
    rel_model.eval()
    assert rel_model(batch, "item").shape == (2, 2)

    with pytest.raises(AssertionError):
        RDL(data=data, col_stats_dict={"user": metadata()}, hidden_dim=4, out_dim=2)
