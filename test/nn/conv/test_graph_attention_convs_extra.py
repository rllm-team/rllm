import torch

from rllm.nn.conv.graph_conv import GTransformerConv, HANConv, HGTConv, RelGNNConv


def hetero_inputs():
    x_dict = {
        "author": torch.randn(3, 8),
        "paper": torch.randn(4, 8),
    }
    edge_index_dict = {
        ("author", "paper"): torch.tensor([[0, 1, 2], [1, 2, 3]]),
        ("paper", "author"): torch.tensor([[0, 2, 3], [0, 1, 2]]),
    }
    return x_dict, edge_index_dict


def test_graph_transformer_conv_homogeneous_and_bipartite_attention():
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    edge_attr = torch.randn(edge_index.size(1), 3)
    conv = GTransformerConv(8, 4, num_heads=2, concat=True, beta=True, edge_dim=3, dropout=0.0)

    out, (attn_edge_index, alpha) = conv(
        torch.randn(3, 8),
        edge_index,
        edge_weight=edge_attr,
        return_attention_weights=True,
    )

    assert out.shape == (3, 8)
    assert torch.equal(attn_edge_index, edge_index)
    assert alpha.shape == (edge_index.size(1), 2)

    bipartite = GTransformerConv((8, 6), 5, num_heads=2, concat=False, root_weight=False)
    out = bipartite((torch.randn(4, 8), torch.randn(2, 6)), torch.tensor([[0, 1, 2], [0, 0, 1]]))

    assert out.shape == (2, 5)


def test_han_conv_outputs_each_node_type_and_semantic_attention():
    torch.manual_seed(0)
    x_dict, edge_index_dict = hetero_inputs()
    metadata = (list(x_dict.keys()), list(edge_index_dict.keys()))
    conv = HANConv(8, 6, metadata, num_heads=2, dropout=0.0)

    out, attn = conv(x_dict, edge_index_dict, return_semantic_attn_weights=True)

    assert out["author"].shape == (3, 6)
    assert out["paper"].shape == (4, 6)
    assert attn["author"].shape == (1,)
    assert attn["paper"].shape == (1,)
    assert str(conv) == "HANConv(6, num_heads=2)"


def test_hgt_conv_outputs_each_node_type():
    torch.manual_seed(0)
    x_dict, edge_index_dict = hetero_inputs()
    metadata = (list(x_dict.keys()), list(edge_index_dict.keys()))
    conv = HGTConv(8, 8, metadata, num_heads=2, dropout=0.0)

    out = conv(x_dict, edge_index_dict)

    assert out["author"].shape == x_dict["author"].shape
    assert out["paper"].shape == x_dict["paper"].shape
    assert str(conv) == "HGTConv(8, num_heads=2)"


def test_relgnn_conv_dim_dim_and_simplified_empty_edges():
    x = torch.randn(4, 8)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
    conv = RelGNNConv("dim-dim", in_dim=8, out_dim=4, num_heads=2, aggr="sum")

    out, (attn_edge_index, alpha) = conv(x, edge_index, return_attention_weights=True)

    assert out.shape == (4, 4)
    assert torch.equal(attn_edge_index, edge_index)
    assert alpha.shape == (edge_index.size(1), 2)

    empty_conv = RelGNNConv("dim-dim", in_dim=8, out_dim=4, num_heads=2, aggr="sum", simplified_MP=True)
    assert empty_conv(x, torch.empty(2, 0, dtype=torch.long)) is None


def test_relgnn_conv_dim_fact_dim_returns_intermediate_source_features():
    src_aggr = torch.randn(5, 8)
    dst_aggr = torch.randn(3, 8)
    dst_attn = torch.randn(2, 8)
    edge_aggr = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 2]])
    edge_attn = torch.tensor([[0, 1, 2], [0, 1, 1]])
    conv = RelGNNConv("dim-fact-dim", in_dim=8, out_dim=4, num_heads=2, aggr="mean")

    out, src_attn = conv((src_aggr, dst_aggr, dst_attn), (edge_attn, edge_aggr))

    assert out.shape == (2, 4)
    assert src_attn.shape == (3, 4)
