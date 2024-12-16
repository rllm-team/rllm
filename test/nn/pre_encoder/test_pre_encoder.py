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
