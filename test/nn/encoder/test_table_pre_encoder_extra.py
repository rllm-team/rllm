import os

import pytest
import torch

from rllm.nn.encoder import (
    HeteroTemporalEncoder,
    ResNetPreEncoder,
    TransTabPreEncoder,
    TromptPreEncoder,
)
from rllm.nn.encoder.table_pre_encoder import TablePreEncoder
from rllm.nn.encoder.col_encoder._embedding_encoder import EmbeddingEncoder
from rllm.types import ColType, StatType


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


class FakeTable:
    def __init__(self, feat_dict, colname_token_ids):
        self.feat_dict = feat_dict
        self.colname_token_ids = colname_token_ids

    def if_materialized(self):
        return True


def basic_metadata():
    return {
        ColType.NUMERICAL: [
            {StatType.MEAN: 0.0, StatType.STD: 1.0},
            {StatType.MEAN: 1.0, StatType.STD: 2.0},
        ],
        ColType.CATEGORICAL: [
            {StatType.COUNT: 3},
        ],
        ColType.BINARY: [
            {StatType.COUNT: 2},
        ],
        ColType.TEXT: [
            {StatType.EMB_DIM: 4},
        ],
        ColType.TIMESTAMP: [
            {
                StatType.YEAR_RANGE: [2020, 2022],
                StatType.MEDIAN_TIME: "2021-01-01 00:00:00",
            }
        ],
    }


def test_table_pre_encoder_validation_and_return_dict():
    with pytest.raises(ValueError):
        TablePreEncoder(
            out_dim=4,
            metadata={ColType.NUMERICAL: [{StatType.MEAN: 0.0, StatType.STD: 1.0}]},
            col_encoder_dict={ColType.NUMERICAL: EmbeddingEncoder()},
        )

    encoder = ResNetPreEncoder(out_dim=4, metadata=basic_metadata())
    feat_dict = {
        ColType.NUMERICAL: torch.tensor([[0.0, 1.0], [2.0, 3.0]]),
        ColType.CATEGORICAL: torch.tensor([[0], [2]]),
        ColType.TEXT: torch.randn(2, 1, 4),
        ColType.TIMESTAMP: torch.tensor(
            [
                [[2020, 0, 0, 0, 0, 0, 0]],
                [[2021, 1, 1, 1, 1, 1, 1]],
            ]
        ),
    }

    out_dict = encoder(feat_dict, return_dict=True)
    out = encoder(feat_dict)

    assert out_dict[ColType.NUMERICAL].shape == (2, 2, 4)
    assert out_dict[ColType.CATEGORICAL].shape == (2, 1, 4)
    assert out_dict[ColType.TEXT].shape == (2, 1, 4)
    assert out_dict[ColType.TIMESTAMP].shape == (2, 1, 4)
    assert out.shape == (2, 5, 4)


def test_trompt_pre_encoder_applies_post_modules():
    encoder = TromptPreEncoder(out_dim=4, metadata=basic_metadata())
    out = encoder(
        {
            ColType.NUMERICAL: torch.tensor([[0.0, 1.0], [2.0, 3.0]]),
            ColType.CATEGORICAL: torch.tensor([[0], [2]]),
        }
    )

    assert out.shape == (2, 3, 4)
    assert torch.isfinite(out).all()


def test_hetero_temporal_encoder_uses_relative_seed_times():
    encoder = HeteroTemporalEncoder(node_types=["user", "item"], channels=4)
    seed_time = torch.tensor([10.0 * 24 * 60 * 60, 20.0 * 24 * 60 * 60])
    time_dict = {
        "user": torch.tensor([9.0 * 24 * 60 * 60, 18.0 * 24 * 60 * 60]),
        "item": torch.tensor([15.0 * 24 * 60 * 60]),
    }
    batch_dict = {"user": torch.tensor([0, 1]), "item": torch.tensor([1])}

    out = encoder(seed_time, time_dict, batch_dict)

    assert out["user"].shape == (2, 4)
    assert out["item"].shape == (1, 4)
    assert torch.isfinite(out["user"]).all()


def test_transtab_pre_encoder_duplicate_columns_and_forward(tmp_path):
    with pytest.raises(ValueError):
        TransTabPreEncoder(
            out_dim=4,
            metadata=basic_metadata(),
            categorical_columns=["shared"],
            numerical_columns=["shared"],
            tokenizer=FakeTokenizer(),
        )

    encoder = TransTabPreEncoder(
        out_dim=4,
        metadata=basic_metadata(),
        categorical_columns=["cat_text"],
        numerical_columns=["num_a", "num_b"],
        binary_columns=["bin_a"],
        tokenizer=FakeTokenizer(),
        use_align_layer=False,
    )
    colname_token_ids = {
        "num_a": (torch.tensor([1, 2]), torch.tensor([1, 1])),
        "num_b": (torch.tensor([3, 4]), torch.tensor([1, 1])),
    }
    table = FakeTable(
        feat_dict={
            ColType.TEXT: (
                torch.tensor([[1, 2, 0], [3, 4, 5]]),
                torch.tensor([[1, 1, 0], [1, 1, 1]]),
            ),
            ColType.NUMERICAL: torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            ColType.BINARY: torch.tensor([[1.0], [0.0]]),
        },
        colname_token_ids=colname_token_ids,
    )

    out = encoder(table)

    assert out["embedding"].shape[0] == 2
    assert out["embedding"].shape[-1] == 4
    assert out["attention_mask"].shape[:1] == (2,)
    with pytest.raises(TypeError):
        encoder({"not": "a table"})
