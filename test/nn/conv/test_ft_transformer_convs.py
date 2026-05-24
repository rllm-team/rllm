import torch

from rllm.nn.conv.table_conv import FTTransformerConv


def test_ft_transformer_convs():
    x = torch.randn(size=(10, 3, 8))
    conv = FTTransformerConv(conv_dim=8, num_heads=2, dropout=0.0)
    x = conv(x)
    assert x.shape == (10, 3, 8)

    conv_cls = FTTransformerConv(conv_dim=8, num_heads=2, dropout=0.0, use_cls=True)
    x_cls = conv_cls(x)

    # The first added column corresponds to CLS token.
    assert x_cls.shape == (10, 8)
