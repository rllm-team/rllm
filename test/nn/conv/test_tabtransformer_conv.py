
import torch

from rllm.nn.conv.tab_transformer_conv import TabTransformerConv
def test_tab_transformer_conv():
    batch_size = 10
    dim = 16
    num_cols = 15
    heads = 8
    dim_head = 16
    # Feature-based embeddings
    x = torch.randn(size=(batch_size, num_cols, dim))
    conv = TabTransformerConv(dim, heads, dim_head, attn_dropout=0., ff_dropout=0.)
    x_out = conv(x)
    assert x_out.shape == (batch_size, num_cols, dim)


