import torch
from rllm.nn.conv.graph_conv import GATConv


def test_gat_conv1():
    n_src = 4
    n_dst = 2
    in_dim = [16, 12]
    out_dim = 8
    num_heads = 2
    edge_dim = 4

    # Feature-based embeddings and adj
    src_nodes = torch.randn(size=(n_src, in_dim[0]))
    dst_nodes = torch.randn(size=(n_dst, in_dim[1]))
    x = (src_nodes, dst_nodes)

    edge_index = torch.tensor(
        [
            [0, 1, 2, 3],
            [0, 0, 1, 1],
        ],
        dtype=torch.long
    )

    edge_attr = torch.randn(size=(edge_index.shape[1], edge_dim))

    conv = GATConv(
        in_dim=in_dim,
        out_dim=out_dim,
        num_heads=num_heads,
        concat=True,
        negative_slope=0.2,
        dropout=0.5,
        edge_dim=edge_dim,
        bias=True,
        residual=True
    )

    assert str(conv) == 'GATConv([16, 12], 8) num_heads=2, concat=True'

    x_out, edge_index, alpha = conv(x, edge_index, edge_attr, return_attention_weights=True)

    assert x_out.shape == (n_dst, out_dim * num_heads)
    assert edge_index.shape == (2, edge_index.shape[1])
    assert alpha.shape == (edge_index.shape[1], num_heads)


def test_gat_conv2():
    n = 4
    in_dim = 16
    out_dim = 8
    num_heads = 2
    edge_dim = 4

    # Feature-based embeddings and adj
    nodes = torch.randn(size=(n, in_dim))
    x = nodes

    edge_index = torch.tensor(
        [
            [0, 1, 2, 3],
            [0, 0, 1, 1],
        ],
        dtype=torch.long
    )

    edge_attr = torch.randn(size=(edge_index.shape[1], edge_dim))

    conv = GATConv(
        in_dim=in_dim,
        out_dim=out_dim,
        num_heads=num_heads,
        concat=True,
        negative_slope=0.2,
        dropout=0.5,
        edge_dim=edge_dim,
        bias=True,
        residual=True
    )

    assert str(conv) == 'GATConv(16, 8) num_heads=2, concat=True'

    x_out, edge_index, alpha = conv(x, edge_index, edge_attr, return_attention_weights=True)

    assert x_out.shape == (n, out_dim * num_heads)
    assert edge_index.shape == (2, edge_index.shape[1])
    assert alpha.shape == (edge_index.shape[1], num_heads)
