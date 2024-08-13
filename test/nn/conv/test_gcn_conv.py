import torch
from rllm.nn.conv import GCNConv


def test_gcn_conv():
    node_size = 4
    in_dim = 16
    out_dim = 8

    # Feature-based embeddings and adj
    x = torch.randn(size=(node_size, in_dim))
    adj = torch.tensor([[1., 1., 1., 1.],
                        [1., 0., 0., 0.],
                        [1., 0., 0., 0.],
                        [1., 0., 0., 0.],])

    conv = GCNConv(in_dim, out_dim)
    assert str(conv) == 'GCNConv(16, 8)'

    x_out = conv(x, adj)
    assert x_out.shape == (node_size, out_dim)
