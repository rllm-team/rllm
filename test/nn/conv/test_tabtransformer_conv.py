import torch

from rllm.types import ColType
from rllm.nn.conv.table_conv import TabTransformerConv


def test_tab_transformer_conv():
    batch_size = 10
    dim = 16
    num_cols = 15
    num_heads = 8
    # Feature-based embeddings
    x = {}
    x[ColType.CATEGORICAL] = torch.randn(size=(batch_size, num_cols, dim))
    x[ColType.NUMERICAL] = torch.randn(size=(batch_size, num_cols, dim))
    conv = TabTransformerConv(dim, num_heads, dropout=0.0)
    x_out = conv(x)
    assert x_out[ColType.CATEGORICAL].shape == (batch_size, num_cols, dim)
    assert x_out[ColType.NUMERICAL].shape == (batch_size, num_cols, dim)
