# This file contains unit tests for pre-encoders used in the rLLM framework.
# The tests ensure that the pre-encoders correctly transform the input features
# as expected.

# The following pre-encoders are tested:
# 1. TabTransformerPreEncoder: Tests the encoding of numerical and categorical
#    features for the TabTransformer model.
# 2. FTTransformerPreEncoder: Tests the encoding of numerical and categorical
#    features for the FTTransformer model.

# Each test function creates a sample DataFrame, initializes the corresponding
# pre-encoder, and verifies the output shapes and values.

import numpy as np
import pandas as pd

from rllm.types import ColType
from rllm.data.table_data import TableData
from rllm.nn.pre_encoder import TabTransformerPreEncoder
from rllm.nn.pre_encoder import FTTransformerPreEncoder


def test_tab_transformer_pre_encoder():
    nodes = 10

    # Create a DataFrame
    df = pd.DataFrame(
        {
            "num_1": np.random.random(nodes),
            "num_2": np.random.random(nodes),
            "cat_1": np.arange(nodes),
            "cat_2": np.arange(nodes),
        },
        dtype=np.float32,
    )
    col_types = {
        "num_1": ColType.NUMERICAL,
        "num_2": ColType.NUMERICAL,
        "cat_1": ColType.CATEGORICAL,
        "cat_2": ColType.CATEGORICAL,
    }
    dataset = TableData(df, col_types, target_col="cat_2")

    # Create and initialize TabTransformerPreEncoder
    pre_encoder = TabTransformerPreEncoder(
        out_dim=1,
        metadata=dataset.metadata,
    )

    # Encode numerical and categorical features
    feat_dict = dataset.get_feat_dict()
    x_emb = pre_encoder(dataset.get_feat_dict())

    # Check the shape of the encoded features
    assert x_emb.shape == (
        nodes,
        feat_dict[ColType.NUMERICAL].size(1) + feat_dict[ColType.CATEGORICAL].size(1),
        1,
    )


def test_ft_transformer_pre_encoder():
    nodes = 10

    # Create a DataFrame
    df = pd.DataFrame(
        {
            "num_1": np.random.random(nodes),
            "num_2": np.random.random(nodes),
            "cat_1": np.arange(nodes),
            "cat_2": np.arange(nodes),
        },
        dtype=np.float32,
    )
    col_types = {
        "num_1": ColType.NUMERICAL,
        "num_2": ColType.NUMERICAL,
        "cat_1": ColType.CATEGORICAL,
        "cat_2": ColType.CATEGORICAL,
    }
    dataset = TableData(df, col_types, target_col="cat_2")

    # Create and initialize FTTransformerPreEncoder
    pre_encoder = FTTransformerPreEncoder(
        out_dim=1,
        metadata=dataset.metadata,
    )

    # Encode numerical and categorical features
    feat_dict = dataset.get_feat_dict()
    x_emb = pre_encoder(dataset.get_feat_dict())

    # Check the shape of the encoded features
    assert x_emb.shape == (
        nodes,
        feat_dict[ColType.NUMERICAL].size(1) + feat_dict[ColType.CATEGORICAL].size(1),
        1,
    )
