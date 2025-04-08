import torch

from rllm.utils import sort_edge_index, index2ptr, _to_csc


def test_sort_edge_index():
    edge_index = torch.tensor([[2, 1, 1, 0],
                               [1, 2, 0, 1]])
    edge_attr = torch.tensor([[1], [2], [3], [4]])

    r = sort_edge_index(edge_index)
    assert torch.equal(r, torch.tensor([[0, 1, 1, 2],
                                        [1, 0, 2, 1]]))

    r = sort_edge_index(edge_index, edge_attr=edge_attr)
    assert len(r) == 2
    assert torch.equal(r[1], torch.tensor([[4],
                                           [3],
                                           [2],
                                           [1]]))


def test_index2ptr():
    index = torch.tensor([0, 1, 1, 2, 2, 3])
    r = index2ptr(index)
    assert torch.equal(r, torch.tensor([0, 1, 3, 5, 6]))


def test_to_csc():
    edge_index = torch.tensor([
        [0, 0, 0, 1, 1, 1, 2, 2, 2],
        [0, 1, 2, 0, 1, 2, 0, 1, 2]
    ], dtype=torch.long)

    col_ptr, row, _ = _to_csc(edge_index)
    assert torch.equal(col_ptr, torch.tensor([0, 3, 6, 9]))
    assert torch.equal(row, torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2]))
