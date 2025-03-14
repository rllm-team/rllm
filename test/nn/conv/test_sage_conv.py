import torch
from rllm.nn.conv.graph_conv import SAGEConv


def test_sage_conv():
    n_src = 4
    n_dst = 2
    in_dim = 16
    out_dim = 8

    # Feature-based embeddings and adj
    src_nodes = torch.randn(size=(n_src, in_dim))
    dst_nodes = torch.randn(size=(n_dst, in_dim))
    x = (src_nodes, dst_nodes)

    edge_index = torch.tensor(
        [
            [0, 1, 2, 3],
            [0, 0, 1, 1],
        ],
        dtype=torch.long
    )

    conv = SAGEConv(in_dim, out_dim, aggr='lstm')
    assert str(conv) == "SAGEConv(16, 8)"

    x_out = conv(x, edge_index)
    assert x_out.shape == (n_dst, out_dim)
