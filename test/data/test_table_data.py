import numpy as np
import pandas as pd
import torch

from rllm.data.table_data import TableData
from rllm.types import ColType


def test_shuffle():
    df = pd.DataFrame({"cat_1": np.arange(10), "cat_2": np.arange(10)})
    col_types = {"cat_1": ColType.CATEGORICAL, "cat_2": ColType.CATEGORICAL}
    dataset = TableData(df, col_types, target_col="cat_2")
    perm = dataset.shuffle(return_perm=True)
    assert torch.equal(torch.from_numpy(dataset.df["cat_1"].values), perm)
    feat = dataset[ColType.CATEGORICAL].view(-1)
    assert torch.equal(feat, perm)


def test_categorical_target_order():
    # Ensures that we do not swap labels in binary classification tasks.
    df = pd.DataFrame({"cat_1": [0, 1, 1, 1], "cat_2": [0, 1, 1, 1]})
    col_types = {"cat_1": ColType.CATEGORICAL, "cat_2": ColType.CATEGORICAL}
    dataset = TableData(df, col_types, target_col="cat_2")

    assert torch.equal(
        dataset[ColType.CATEGORICAL],
        torch.tensor([[0], [1], [1], [1]]),
    )
    assert torch.equal(dataset.y, torch.tensor([0, 1, 1, 1]))
