from __future__ import annotations

import torch

from rllm.nn.conv.graph_conv import GATConv, GCNConv
from rllm.nn.conv.table_conv import FTTransformerConv


def test_gcn_conv_aggregates_source_rows_into_destinations_and_keeps_isolated_rows():
    conv = GCNConv(2, 2, bias=False, normalize=False)
    with torch.no_grad():
        conv.linear.weight.copy_(torch.eye(2))
    features = torch.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    edge_index = torch.tensor([[0, 1], [1, 1]], dtype=torch.long)

    output = conv(features, edge_index, dim_size=3)

    assert torch.equal(
        output, torch.tensor([[0.0, 0.0], [3.0, 30.0], [0.0, 0.0]])
    )


def test_gat_attention_normalizes_per_destination_and_backpropagates():
    torch.manual_seed(7)
    conv = GATConv(
        (3, 3), out_dim=2, num_heads=2, concat=False, dropout=0.0, bias=False
    )
    source = torch.randn(4, 3, requires_grad=True)
    destination = torch.randn(3, 3, requires_grad=True)
    edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 2]], dtype=torch.long)

    output, (_, attention) = conv(
        (source, destination), edge_index, return_attention_weights=True
    )

    assert output.shape == (3, 2)
    assert attention.shape == (4, 2)
    for destination_id in range(3):
        mask = edge_index[1] == destination_id
        assert torch.allclose(attention[mask].sum(dim=0), torch.ones(2), atol=1e-6)
    output.square().sum().backward()
    assert source.grad is not None and torch.isfinite(source.grad).all()
    assert destination.grad is not None and torch.isfinite(destination.grad).all()


def test_ft_transformer_output_modes_and_gradients_are_finite():
    torch.manual_seed(11)
    inputs = torch.randn(2, 3, 4, requires_grad=True)
    token_model = FTTransformerConv(
        conv_dim=4, num_heads=2, dropout=0.0, use_cls=False
    )
    cls_model = FTTransformerConv(
        conv_dim=4, num_heads=2, dropout=0.0, use_cls=True
    )
    token_model.eval()
    cls_model.eval()

    token_output = token_model(inputs)
    cls_output = cls_model(inputs)

    assert token_output.shape == (2, 3, 4)
    assert cls_output.shape == (2, 4)
    (token_output.sum() + cls_output.sum()).backward()
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
