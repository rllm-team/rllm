import pytest
import torch

from rllm.nn.conv.graph_conv import (
    AddAggregator,
    MaxAggregator,
    MeanAggregator,
    MessagePassing,
    MinAggregator,
    ProdAggregator,
    SumAggregator,
)


def test_basic_aggregators_reduce_by_index():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    index = torch.tensor([0, 0, 1])

    assert torch.equal(SumAggregator()(x, index, dim_size=3), torch.tensor([[4.0, 6.0], [5.0, 6.0], [0.0, 0.0]]))
    assert torch.equal(AddAggregator()(x, index, dim_size=2), torch.tensor([[4.0, 6.0], [5.0, 6.0]]))
    assert torch.equal(MeanAggregator()(x, index, dim_size=2), torch.tensor([[2.0, 3.0], [5.0, 6.0]]))
    assert torch.equal(MaxAggregator()(x, index, dim_size=2), torch.tensor([[3.0, 4.0], [5.0, 6.0]]))
    assert torch.equal(MinAggregator()(x, index, dim_size=2), torch.tensor([[1.0, 2.0], [5.0, 6.0]]))
    assert torch.equal(ProdAggregator()(x, index, dim_size=2), torch.tensor([[3.0, 8.0], [5.0, 6.0]]))


def test_aggregator_dense_batch_padding_and_filtering():
    aggr = SumAggregator()
    x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    batch = torch.tensor([0, 0, 0, 1])

    dense, mask = aggr.to_dense_batch(x, batch, max_num_nodes=2, fill_value=-1.0)

    assert dense.tolist() == [[[1.0], [2.0]], [[4.0], [-1.0]]]
    assert mask.tolist() == [[True, True], [True, False]]


def test_message_passing_sum_with_edge_index_and_sparse_weights():
    conv = MessagePassing(aggr="sum")
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    edge_index = torch.tensor([[0, 1, 2], [1, 1, 0]])

    out = conv.propagate(x, edge_index, dim_size=3)

    assert torch.equal(out, torch.tensor([[5.0, 6.0], [4.0, 6.0], [0.0, 0.0]]))

    sparse_adj = torch.sparse_coo_tensor(edge_index, torch.tensor([0.5, 2.0, 1.0]), (3, 3))
    weighted = conv.propagate(x, sparse_adj, dim_size=3)

    assert torch.equal(weighted, torch.tensor([[5.0, 6.0], [6.5, 9.0], [0.0, 0.0]]))


def test_message_passing_retrieve_feats_and_error_paths():
    conv = MessagePassing(aggr="mean")
    x = torch.arange(12).view(4, 3)
    edge_index = torch.tensor([[0, 2, 3], [1, 1, 0]])

    src, dst = conv.retrieve_feats(x, edge_index)

    assert torch.equal(src, x[[0, 2, 3]])
    assert torch.equal(dst, x[[1, 1, 0]])
    assert torch.equal(conv.retrieve_feats((x, x + 100), edge_index, dim=1), (x + 100)[[1, 1, 0]])
    with pytest.raises(ValueError):
        MessagePassing(aggr="missing")
    with pytest.raises(ValueError):
        conv.propagate(None, edge_index)
    with pytest.raises(ValueError):
        conv.propagate(x, torch.ones(2, 2, 2))
