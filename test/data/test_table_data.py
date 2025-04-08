import os
import os.path as osp

import numpy as np
import pandas as pd
import torch

from rllm.types import ColType
from rllm.datasets import TML1MDataset
from rllm.data.table_data import TableData


def test_table_data():
    df = pd.DataFrame({"cat_1": [0, 1, 2, 3],
                       "cat_2": [4, 5, 6, 7],
                       "cat_3": [10, 20, 30, 40]})
    col_types = {"cat_1": ColType.CATEGORICAL,
                 "cat_2": ColType.NUMERICAL,
                 "cat_3": ColType.NUMERICAL}
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
