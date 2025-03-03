import sys

import torch

sys.path.append("./")
from rllm.nn.conv.graph_conv import GCNConv

torch.manual_seed(0)


def dense_to_sparse(adj):
    indices = torch.nonzero(adj).t()
    values = adj[indices[0], indices[1]]
    return torch.sparse_coo_tensor(indices, values, adj.size())


def test_gcn_conv():
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

    edge_index = torch.nonzero(adj).t()
    # adj = dense_to_sparse(adj)

    conv = GCNConv(in_dim, out_dim)
    assert str(conv) == "GCNConv(16, 8)"

    x_out = conv(x, edge_index)
    assert x_out.shape == (node_size, out_dim)


test_gcn_conv()
