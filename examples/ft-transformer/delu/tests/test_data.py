import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

import delu.data
import delu.utils.data

from .util import ignore_deprecated_warning


def test_enumerate():
    dataset = TensorDataset(torch.arange(10), torch.arange(10))
    x = delu.utils.data.Enumerate(dataset)
    assert x.dataset is dataset
    assert len(x) == 10
    assert x[3] == (3, (torch.tensor(3), torch.tensor(3)))


def test_fndataset():
    dataset = delu.data.FnDataset(lambda x: x * 2, 3)
    assert len(dataset) == 3
    assert dataset[0] == 0
    assert dataset[1] == 2
    assert dataset[2] == 4

    dataset = delu.data.FnDataset(lambda x: x * 2, 3, lambda x: x * 3)
    assert len(dataset) == 3
    assert dataset[0] == 0
    assert dataset[1] == 6
    assert dataset[2] == 12

    dataset = delu.data.FnDataset(lambda x: x * 2, [1, 10, 100])
    assert len(dataset) == 3
    assert dataset[0] == 2
    assert dataset[1] == 20
    assert dataset[2] == 200

    dataset = delu.data.FnDataset(lambda x: x * 2, (x for x in range(0, 10, 4)))
    assert len(dataset) == 3
    assert dataset[0] == 0
    assert dataset[1] == 8
    assert dataset[2] == 16


def test_index_dataset():
    n = 10
    d = delu.utils.data.IndexDataset(n)
    assert len(d) == n
    for i in range(n):
        assert d[i] == i
    with pytest.raises(IndexError):
        d[-1]
    with pytest.raises(IndexError):
        d[n]


@ignore_deprecated_warning
def test_collate():
    # just test that the function is still a valid alias
    assert torch.equal(delu.data.collate([1])[0], torch.tensor(1))


@ignore_deprecated_warning
def test_iloader():
    with pytest.raises(AssertionError):
        delu.data.IndexLoader(0)

    for x in range(1, 10):
        assert len(delu.data.IndexLoader(x)) == x

    data = torch.arange(10)
    for batch_size in range(1, len(data) + 1):
        torch.manual_seed(batch_size)
        correct = list(DataLoader(data, batch_size, shuffle=True, drop_last=True))
        torch.manual_seed(batch_size)
        actual = list(
            delu.data.IndexLoader(len(data), batch_size, shuffle=True, drop_last=True)
        )
        for x, y in zip(actual, correct):
            assert torch.equal(x, y)
