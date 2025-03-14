import torch

from rllm.nn.conv.table_conv import FTTransformerConv


def test_ft_transformer_convs():
    x = torch.randn(size=(10, 3, 8))
    conv = FTTransformerConv(dim=8)
    x = conv(x)
    assert x.shape == (10, 3, 8)

    conv_cls = FTTransformerConv(dim=8, use_cls=True)
    x_cls = conv_cls(x)

    # The first added column corresponds to CLS token.
    assert x_cls.shape == (10, 8)
