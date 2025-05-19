import os
import os.path as osp

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from rllm.types import ColType
from rllm.datasets import TML1MDataset
from rllm.data.table_data import TableData, TextEmbedderConfig


def test_table_data():
    df = pd.DataFrame(
        {"cat_1": [0, 1, 2, 3], "cat_2": [4, 5, 6, 7], "cat_3": [10, 20, 30, 40]}
    )
    col_types = {
        "cat_1": ColType.CATEGORICAL,
        "cat_2": ColType.NUMERICAL,
        "cat_3": ColType.NUMERICAL,
    }
    table = TableData(df, col_types, target_col="cat_2")

    # test lazy materialize
    table.lazy_materialize(keep_df=True)
    assert len(table) == 4

    # test copy
    import copy

    n_table = copy.copy(table)
    n_table.feat_dict = None
    n_table.df = None
    assert table.feat_dict is not None and n_table.feat_dict is None
    assert table.df is not None and n_table.df is None

    # test Tensor slice
    n_table = table[torch.tensor([0, 2])]
    assert len(table) == 4 and len(n_table) == 2
    assert id(n_table.df) == id(table.df)
    assert id(n_table.metadata) == id(table.metadata)

    assert hasattr(n_table, "y")
    assert n_table.y.numel() == 2


def test_shuffle():
    df = pd.DataFrame({"cat_1": np.arange(10), "cat_2": np.arange(10)})
    col_types = {"cat_1": ColType.CATEGORICAL, "cat_2": ColType.CATEGORICAL}
    dataset = TableData(df, col_types, target_col="cat_2")
    dataset.lazy_materialize()
    perm = dataset.shuffle(return_perm=True)
    assert torch.equal(torch.from_numpy(dataset.df["cat_1"].values), perm)
    feat = dataset[ColType.CATEGORICAL].view(-1)
    assert torch.equal(feat, perm)


def test_categorical_target_order():
    # Ensures that we do not swap labels in binary classification tasks.
    df = pd.DataFrame({"cat_1": [0, 1, 1, 1], "cat_2": [0, 1, 1, 1]})
    col_types = {"cat_1": ColType.CATEGORICAL, "cat_2": ColType.CATEGORICAL}
    dataset = TableData(df, col_types, target_col="cat_2")
    dataset.lazy_materialize()

    assert torch.equal(
        dataset[ColType.CATEGORICAL],
        torch.tensor([[0], [1], [1], [1]]),
    )
    assert torch.equal(dataset.y, torch.tensor([0, 1, 1, 1]))


def test_compatiblity():
    root = osp.dirname(osp.dirname(osp.dirname(__file__)))
    data_path = osp.join(root, "data")
    dataset = TML1MDataset(cached_dir=data_path, force_reload=True)
    user_table, movie_table, rating_table, _ = dataset.data_list
    rating_table.table_name = "rating_table"
    user_table.table_name = "user_table"
    movie_table.table_name = "movie_table"
    rating_table.fkeys = ["UserID", "MovieID"]
    assert rating_table.fkeys == ["UserID", "MovieID"]
    cache_dir = "./cached_dir"
    os.makedirs(cache_dir, exist_ok=True)
    user_table.save(osp.join(cache_dir, "user_table.pt"))
    user_table = TableData.load(osp.join(cache_dir, "user_table.pt"))
    os.remove(osp.join(cache_dir, "user_table.pt"))
    os.rmdir(cache_dir)
    assert user_table.table_name == "user_table"


def test_text_embedding():
    csv_content = [
        ["Column1", "Column2", "Column3", "Column4", "Column5", "Column6"],
        ["Value1", "Value2", "22", "1", '"hello"', "Value6"],
        ["Value7", "Value8", "355", "2", '"this is"', "Value12"],
        ["Value13", "Value14", "67", "35", '"a test"', "Value18"],
        ["Value19", "Value20", "88", "64", '"thanks for your attention!"', "Value24"],
    ]
    df = pd.DataFrame(csv_content[1:], columns=csv_content[0])

    class embedder:
        def __init__(self, model_name="F:\\wenjian\\learn\\Github\\all-MiniLM-L6-v2"):
            self.model = SentenceTransformer(model_name)

        def __call__(self, texts):
            return torch.tensor(self.model.encode(texts))

    col_types = {
        "Column1": ColType.CATEGORICAL,
        "Column2": ColType.CATEGORICAL,
        "Column3": ColType.NUMERICAL,
        "Column4": ColType.NUMERICAL,
        "Column5": ColType.TEXT,
        "Column6": ColType.TEXT,
    }
    cfg = TextEmbedderConfig(text_embedder=embedder(), batch_size=8)
    data = TableData(
        df=df, col_types=col_types, target_col="Survived", text_embedder_config=cfg
    )
    assert data.feat_dict[ColType.TEXT].shape == (4, 2, 384)
