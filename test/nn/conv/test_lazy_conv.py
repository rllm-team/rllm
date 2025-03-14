import torch
from rllm.nn.conv.graph_conv import LazyConv


def normalize_adj(adj):
    adj = adj + torch.eye(adj.size(0))
    rowsum = adj.sum(1)  # [4, 2, 2, 2]
    d_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    return d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt


def test_gcn_conv():
    num_nodes = 4
    feats_size = 8

    # Feature-based embeddings and adj
    x = torch.randn(size=(num_nodes, feats_size))
    adj = torch.tensor(
        [
            [0.0, 1.0, 1.0, 1.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ]
    )
    adj = normalize_adj(adj)
    adj = adj.to_sparse()

    conv = LazyConv(0.1)
    assert str(conv) == "LazyConv(beta: 0.1)"

    x_out = conv(x, adj)
    assert x_out.shape == x.shape
