import copy

import numpy as np
import pandas as pd
import pytest
import torch

from rllm.data.table_data import TableData, TableDataset
from rllm.preprocessing.text_tokenize import TokenizerConfig
from rllm.preprocessing.word_embedding import TextEmbedderConfig
from rllm.types import ColType, StatType, TableType, TaskType


def make_basic_table(lazy_feature=False):
    df = pd.DataFrame(
        {
            "id": [10, 11, 12, 13],
            "cat": ["a", "b", "a", "b"],
            "num": [1.0, 2.0, 3.0, 4.0],
            "target": [0, 1, 0, 1],
        }
    )
    return TableData(
        df,
        col_types={
            "cat": ColType.CATEGORICAL,
            "num": ColType.NUMERICAL,
            "target": ColType.CATEGORICAL,
        },
        pkey="id",
        target_col="target",
        lazy_feature=lazy_feature,
    )


def test_table_data_materialize_slice_properties_and_copy():
    table = make_basic_table(lazy_feature=True)

    assert table.index_col == "id"
    assert table.cols == ["id", "cat", "num", "target"]
    assert table.feat_cols == ["cat", "num"]
    assert table.task_type == TaskType.BINARY_CLASSIFICATION
    assert table.count_numerical_features() == ["num"]
    assert table.count_categorical_features() == {"cat": 2}
    assert not table.if_materialized()

    with pytest.raises(ValueError):
        table[ColType.NUMERICAL]

    table.lazy_materialize(keep_df=True)
    assert table.if_materialized()
    assert len(table) == 4
    assert table.num_rows == 4
    assert table.num_cols == 2
    assert table[ColType.NUMERICAL].shape == (4, 1)
    assert table.y.tolist() == [0, 1, 0, 1]
    assert StatType.MEAN in table.metadata[ColType.NUMERICAL][0]

    sliced = table[torch.tensor([0, 2])]
    assert len(sliced) == 2
    assert sliced.y.tolist() == [0, 0]
    assert sliced.df is table.df
    assert sliced.metadata is table.metadata

    copied = copy.copy(table)
    copied.feat_dict = None
    assert table.feat_dict is not None
    assert copied.feat_dict is None


def test_table_data_get_feat_dict_dataset_dataloader_and_masks():
    table = make_basic_table()

    first_half = table.get_feat_dict(0, 2)
    middle_half = table.get_feat_dict(0.25, 0.75)
    mask = torch.tensor([True, False, True, False])
    masked = table.get_feat_dict_from_mask(mask)
    train_ds, val_ds, test_ds = table.get_dataset(0.5, 0.25, 0.25)
    mask_datasets = table.get_dataset_from_mask(mask, ~mask, torch.tensor([False, True, False, True]))
    loaders = table.get_dataloader(2, 1, 1, batch_size=1, shuffle=False)

    assert first_half[ColType.NUMERICAL].shape == (2, 1)
    assert middle_half[ColType.CATEGORICAL].shape == (2, 1)
    assert masked[ColType.NUMERICAL].shape == (2, 1)
    assert [len(ds) for ds in (train_ds, val_ds, test_ds)] == [2, 1, 1]
    assert [len(ds) for ds in mask_datasets] == [2, 2, 2]
    assert len(list(loaders[0])) == 2

    with pytest.raises(AssertionError):
        table.get_feat_dict(0, 0.5)
    with pytest.raises(AssertionError):
        table.get_dataset(0.2, 0.2, 0.2)


def test_table_dataset_returns_feature_row_and_label():
    feat_dict = {ColType.NUMERICAL: torch.tensor([[1.0], [2.0]])}
    y = torch.tensor([0, 1])
    dataset = TableDataset(feat_dict, y)

    features, label = dataset[1]

    assert len(dataset) == 2
    assert features[ColType.NUMERICAL].tolist() == [2.0]
    assert label.item() == 1


def test_table_data_shuffle_save_load_fkeys_and_removed_dataframe(tmp_path):
    table = TableData(
        pd.DataFrame(
            {
                "id": [0, 1, 2, 3],
                "fk": [100, 101, 100, 102],
                "cat": [0, 1, 0, 1],
                "target": [1, 0, 1, 0],
            }
        ),
        {"fk": ColType.CATEGORICAL, "cat": ColType.CATEGORICAL, "target": ColType.CATEGORICAL},
        pkey="id",
        fkeys=["fk"],
        target_col="target",
    )

    assert table.fkeys == ["fk"]
    assert table.fkey_index("fk").tolist() == [100, 101, 100, 102]

    perm = table.shuffle(return_perm=True)
    assert sorted(perm.tolist()) == [0, 1, 2, 3]
    assert table[ColType.CATEGORICAL].shape == (4, 2)

    save_path = tmp_path / "table.pt"
    table.save(save_path)
    loaded = TableData.load(save_path)
    assert loaded.table_name == table.table_name
    assert loaded.fkeys == ["fk"]

    table.lazy_materialize(keep_df=False)
    with pytest.raises(ValueError):
        _ = table.cols
    with pytest.raises(ValueError):
        table.fkeys = ["fk"]


def test_table_data_pkey_validation_and_table_type_inference():
    with pytest.raises(AssertionError):
        TableData(
            pd.DataFrame({"id": [1, 1], "x": [1, 2]}),
            {"x": ColType.NUMERICAL},
            pkey="id",
        )

    with pytest.raises(AssertionError):
        TableData(
            pd.DataFrame({"x": [1, 2]}),
            {"x": ColType.NUMERICAL},
            pkey="missing",
        )

    relationship = TableData(
        pd.DataFrame({"src": [1, 2], "dst": [3, 4], "value": [5, 6]}),
        {"src": ColType.CATEGORICAL, "dst": ColType.CATEGORICAL, "value": ColType.NUMERICAL},
        fkeys=["src", "dst"],
        lazy_feature=True,
    )
    assert relationship.table_type == TableType.RELATIONSHIPTABLE


def test_text_embedding_and_tokenized_text_metadata_paths():
    df = pd.DataFrame({"text": ["aa", "bbb"], "num": [1, 2], "target": [0, 1]})

    def embedder(texts):
        return torch.tensor([[len(text), len(text) + 1] for text in texts], dtype=torch.float32)

    embedded = TableData(
        df.copy(),
        {"text": ColType.TEXT, "num": ColType.NUMERICAL, "target": ColType.CATEGORICAL},
        target_col="target",
        text_embedder_config=TextEmbedderConfig(text_embedder=embedder),
    )
    assert embedded.feat_dict[ColType.TEXT].shape == (2, 1, 2)
    assert embedded.metadata[ColType.TEXT][0][StatType.EMB_DIM] == 2

    tokenizer_config = TokenizerConfig(
        tokenizer=lambda texts: {"input_ids": [[len(text), 0] for text in texts]},
        tokenize_combine=True,
        include_colname=False,
        save_colname_token_ids=True,
    )
    tokenized = TableData(
        df.copy(),
        {"text": ColType.TEXT, "num": ColType.NUMERICAL, "target": ColType.CATEGORICAL},
        target_col="target",
        tokenizer_config=tokenizer_config,
    )
    ids, mask = tokenized.feat_dict[ColType.TEXT]
    assert ids.tolist() == [[2, 0], [3, 0]]
    assert mask.tolist() == [[1, 0], [1, 0]]
    assert set(tokenized.colname_token_ids) == {"text", "num", "target"}


def test_convert_text_coltypes_and_numeric_regression_task():
    table = TableData(
        pd.DataFrame({"cat": ["a", "b"], "target": [1.5, 2.5]}),
        {"cat": ColType.CATEGORICAL, "target": ColType.NUMERICAL},
        target_col="target",
        convert_text_coltypes={ColType.CATEGORICAL},
        text_embedder_config=TextEmbedderConfig(
            text_embedder=lambda texts: torch.ones((len(texts), 3))
        ),
    )

    assert table.col_types["cat"] == ColType.TEXT
    assert table.task_type == TaskType.REGRESSION
    assert table.num_classes == 2
