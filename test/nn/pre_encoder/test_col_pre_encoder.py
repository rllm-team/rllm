# This file contains unit tests for column pre-encoders used in the rLLM
# framework. The tests ensure that the pre-encoders correctly transform the
# input features as expected.

# The following pre-encoders are tested:
# 1. ReshapeEncoder: Tests the reshaping of numerical and categorical features.
# 2. EmbeddingEncoder: Placeholder for testing the embedding of categorical features.
# 3. LinearEncoder: Placeholder for testing the linear transformation of numerical features.

# Each test function creates a sample DataFrame, initializes the corresponding
# pre-encoder, and verifies the output shapes and values.

import numpy as np
import pandas as pd

import torch

from rllm.types import ColType
from rllm.data.table_data import TableData
from rllm.nn.pre_encoder._reshape_encoder import ReshapeEncoder
from rllm.nn.pre_encoder._embedding_encoder import EmbeddingEncoder
from rllm.nn.pre_encoder._linear_encoder import LinearEncoder


def test_reshape_encoder():
    df = pd.DataFrame(
        {
            "num_1": np.random.random(10),
            "num_2": np.random.random(10),
            "cat_1": np.arange(10),
            "cat_2": np.arange(10),
            "cat_3": np.arange(10),
        },
        dtype=np.float32,
    )
    col_types = {
        "num_1": ColType.NUMERICAL,
        "num_2": ColType.NUMERICAL,
        "cat_1": ColType.CATEGORICAL,
        "cat_2": ColType.CATEGORICAL,
        "cat_3": ColType.CATEGORICAL,
    }
    dataset = TableData(df, col_types, target_col="cat_3")

    pre_encoder = ReshapeEncoder()
    pre_encoder.post_init()

    x_num = dataset.get_feat_dict()[ColType.NUMERICAL].clone()
    x_cat = dataset.get_feat_dict()[ColType.CATEGORICAL].clone()
    x_num_emb = pre_encoder(x_num)
    x_cat_emb = pre_encoder(x_cat)

    # Check the shape of the encoded features
    assert x_num_emb.shape == (x_num.size(0), x_num.size(1), 1)
    assert x_cat_emb.shape == (x_cat.size(0), x_cat.size(1), 1)

    # Make sure other column embeddings are unchanged
    assert torch.allclose(x_num, dataset.get_feat_dict()[ColType.NUMERICAL])
    assert torch.allclose(x_cat, dataset.get_feat_dict()[ColType.CATEGORICAL])


def test_embedding_encoder():
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
    pre_encoder = EmbeddingEncoder(
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


def test_linear_encoder():
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
    pre_encoder = LinearEncoder(
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
