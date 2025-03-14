import torch
from rllm.nn.conv.graph_conv import GCNConv


def test_gcn_conv1():
    node_size = 4
    in_dim = 16
    out_dim = 8

    # Feature-based embeddings and adj
    x = torch.randn(size=(node_size, in_dim))
    adj = torch.tensor(
        [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ]
    )
    adj = adj.to_sparse()

    conv = GCNConv(in_dim, out_dim)
    assert str(conv) == "GCNConv(16, 8)"

    x_out = conv(x, adj)
    assert x_out.shape == (node_size, out_dim)


def test_gcn_conv2():
    node_size = 4
    in_dim = 16
    out_dim = 8

    # Feature-based embeddings and adj
    x = torch.randn(size=(node_size, in_dim))
    edge_index = torch.tensor(
        [
            [0, 1, 2, 3],
            [0, 0, 1, 1],
        ],
        dtype=torch.long
    )

    conv = GCNConv(in_dim, out_dim)
    assert str(conv) == "GCNConv(16, 8)"

    x_out = conv(x, edge_index)
    assert x_out.shape == (node_size, out_dim)
