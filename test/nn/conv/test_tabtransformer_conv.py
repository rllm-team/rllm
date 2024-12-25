import torch

from rllm.nn.conv.table_conv import TabTransformerConv


def test_tab_transformer_conv():
    batch_size = 10
    dim = 16
    num_cols = 15
    heads = 8
    head_dim = 16
    # Feature-based embeddings
    x = torch.randn(size=(batch_size, num_cols, dim))
    conv = TabTransformerConv(dim, heads, head_dim, attn_dropout=0.0, ff_dropout=0.0)
    x_out = conv(x)
    assert x_out.shape == (batch_size, num_cols, dim)
