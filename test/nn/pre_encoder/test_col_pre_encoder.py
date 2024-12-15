import numpy as np
import pandas as pd
import torch

from rllm.types import ColType
from rllm.data.table_data import TableData
from rllm.nn.pre_encoder import EmbeddingPreEncoder, LinearPreEncoder


def test_embedding_pre_encoder():
    df = pd.DataFrame(
        {
            "cat_1": np.arange(10),
            "cat_2": np.arange(10),
            "cat_3": np.arange(10),
        },
        dtype=np.int64,
    )
    col_types = {
        "cat_1": ColType.CATEGORICAL,
        "cat_2": ColType.CATEGORICAL,
        "cat_3": ColType.CATEGORICAL,
    }
    dataset = TableData(df, col_types, target_col="cat_3")
    pre_encoder = EmbeddingPreEncoder(
        out_dim=4,
        stats_list=dataset.metadata[ColType.CATEGORICAL],
    )
    pre_encoder.post_init()
    x_cat = dataset.get_feat_dict()[ColType.CATEGORICAL].clone()
    x_emb = pre_encoder(x_cat)
    assert x_emb.shape == (x_cat.size(0), x_cat.size(1), 4)
    assert torch.allclose(x_cat, dataset.get_feat_dict()[ColType.CATEGORICAL])

    # Perturb the first column
    x_cat[:, 0] = x_cat[:, 0] + 1
    x_perturbed = pre_encoder(x_cat)
    # Make sure other column embeddings are unchanged
    assert (x_perturbed[:, 1:, :] == x_emb[:, 1:, :]).all()


def test_linear_pre_encoder():
    df = pd.DataFrame(
        {
            "num_1": np.random.random(10),
            "num_2": np.random.random(10),
            "num_3": np.random.random(10),
        },
        dtype=np.float32,
    )
    col_types = {
        "num_1": ColType.NUMERICAL,
        "num_2": ColType.NUMERICAL,
        "num_3": ColType.NUMERICAL,
    }
    dataset = TableData(df, col_types, target_col="num_3")
    pre_encoder = LinearPreEncoder(
        out_dim=4,
        stats_list=dataset.metadata[ColType.NUMERICAL],
    )
    pre_encoder.post_init()
    x_num = dataset.get_feat_dict()[ColType.NUMERICAL].clone()
    x_emb = pre_encoder(x_num)
    assert x_emb.shape == (x_num.size(0), x_num.size(1), 4)
    assert torch.allclose(x_num, dataset.get_feat_dict()[ColType.NUMERICAL])

    # Perturb the first column
    x_num[:, 0] = x_num[:, 0] + 42.0
    x_perturbed = pre_encoder(x_num)
    # Make sure other column embeddings are unchanged
    assert (x_perturbed[:, 1:, :] == x_emb[:, 1:, :]).all()
