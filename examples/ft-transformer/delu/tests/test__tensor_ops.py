import dataclasses
from collections.abc import Mapping, Sequence
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import delu

from .util import Point, PointDC


def flatten(data):
    if data is None or isinstance(data, (bool, int, float, str, bytes, torch.Tensor)):
        yield data
    elif isinstance(data, nn.Module):
        yield from data.parameters()
    elif isinstance(data, Sequence):
        for x in data:
            yield from flatten(x)
    elif isinstance(data, Mapping):
        for x in data.values():
            yield from flatten(x)
    elif dataclasses.is_dataclass(data):
        for x in dataclasses.fields(data):
            yield from flatten(getattr(data, x.name))
    else:
        for x in vars(data).values():
            yield from flatten(x)


def test_to():
    f32 = torch.float32
    f16 = torch.float16

    pods = [None, False, -1231231231321312321313, 4.567, 'aaaa', b'bbbbbbbb']
    for x in pods:
        assert delu.to(x, f16) is x
    for x, y in zip(pods, delu.to(pods, f16)):
        assert x is y

    assert delu.to(pods, f32) is not pods
    assert delu.to(tuple(pods), f32) is not pods
    assert delu.to(dict(enumerate(pods)), f32) is not pods

    def make(dtype):
        return torch.randn(2, 3, dtype=dtype)

    for dtype in f32, f16:
        x = make(dtype)
        assert delu.to(x, dtype) is x
    assert delu.to(make(f32), f16).dtype is f16

    for Container in tuple, Point, list:
        constructor = Container._make if Container is Point else Container
        for dtype in [f32, f16]:
            x = constructor([make(f32), make(f32)])
            out = delu.to(x, dtype)
            assert isinstance(out, Container)
            assert all(x.dtype is dtype for x in out)
            if dtype is f32:
                for x, y in zip(out, x):
                    assert x is y

    data = [make(f32), make(f32)]
    for x, y in zip(delu.to(data, f32), data):
        assert x is y
    assert all(x.dtype is f16 for x in delu.to(data, f16))

    @dataclasses.dataclass
    class A:
        a: torch.Tensor

    data = {
        'a': [make(f32), (make(f32), make(f32))],
        'b': {'c': {'d': [[[make(f32)]]]}},
        'c': Point(make(f32), {'d': make(f32)}),
        'f': A(a=make(f32)),
        'g': SimpleNamespace(layer1=[nn.LayerNorm(3)], layer2=nn.Linear(2, 2)),
    }
    for x, y in zip(flatten(delu.to(data, f32)), flatten(data)):
        assert x is y
    for x, y in zip(flatten(delu.to(data, f16)), flatten(data)):
        assert type(x) is type(y)
        if isinstance(x, torch.Tensor):
            assert x.dtype is f16


def test_cat():
    # The function is mostly tested in the doctests.
    with pytest.raises(ValueError):
        delu.cat([])

    with pytest.raises(ValueError):
        delu.cat([0])

    with pytest.raises(ValueError):
        delu.cat([[0], [1]])

    sequence = [
        Point(torch.tensor([[0, 1]]), torch.tensor([[[0.0, 1.0]]])),
        Point(torch.tensor([[2, 3]]), torch.tensor([[[2.0, 3.0]]])),
    ]

    actual0 = delu.cat(sequence)
    assert isinstance(actual0, Point)
    assert torch.equal(actual0.x, torch.tensor([[0, 1], [2, 3]]))
    assert torch.equal(actual0.y, torch.tensor([[[0.0, 1.0]], [[2.0, 3.0]]]))

    actual1 = delu.cat(sequence, dim=1)
    assert isinstance(actual1, Point)
    assert torch.equal(actual1.x, torch.tensor([[0, 1, 2, 3]]))
    assert torch.equal(actual1.y, torch.tensor([[[0.0, 1.0], [2.0, 3.0]]]))


@pytest.mark.parametrize('batch_size', list(range(1, 11)))
def test_iter_batches(batch_size):
    def check(batches, correct):
        sizes = list(map(len, batches))
        assert sum(sizes) == len(correct)
        assert (
            set(sizes) == {batch_size}
            or set(sizes[:-1]) == {batch_size}
            and sizes[-1] == len(correct) % batch_size
        )
        assert torch.equal(torch.cat(batches), correct)

    n = 10

    # test batch size only
    data = torch.arange(n)
    batches = list(delu.iter_batches(data, batch_size))
    check(batches, data)

    data = Point(torch.arange(n), torch.arange(n, n * 2))
    batches = list(delu.iter_batches(data, batch_size))
    assert all(isinstance(t, Point) for t in batches)
    for i in range(2):
        batches_i = list(x[i] for x in batches)
        check(batches_i, data[i])

    data = (torch.arange(n), torch.arange(n, n * 2))
    batches = list(delu.iter_batches(data, batch_size))
    for i in range(2):
        batches_i = list(x[i] for x in batches)
        check(batches_i, data[i])

    data = {'a': torch.arange(n), 'b': torch.arange(n, n * 2)}
    batches = list(delu.iter_batches(data, batch_size))
    for key in data:
        batches_i = list(x[key] for x in batches)
        check(batches_i, data[key])

    assert (
        len(list(delu.iter_batches(torch.arange(n), batch_size, drop_last=True)))
        == n // batch_size
    )


def test_iter_batches_shuffle():
    a = torch.arange(1000)

    with pytest.raises(ValueError):
        next(delu.iter_batches(a, 10, generator=torch.Generator()))

    batches = list(delu.iter_batches(a, 10, shuffle=True))
    assert not torch.equal(torch.cat(batches), a)
    assert sorted(torch.cat(batches).tolist()) == a.tolist()

    gen = torch.Generator()
    state = gen.get_state()
    batches0 = list(delu.iter_batches(a, 10, shuffle=True, generator=gen))
    gen.set_state(state)
    batches1 = list(delu.iter_batches(a, 10, shuffle=True, generator=gen))
    for x in zip(batches0, batches1):
        assert torch.equal(*x)


def test_iter_batches_bad_input():
    # Empty input.
    @dataclasses.dataclass
    class Empty:
        pass

    with pytest.raises(ValueError):
        next(delu.iter_batches(torch.empty(0), 1))
    with pytest.raises(ValueError):
        next(delu.iter_batches((), 1))
    with pytest.raises(ValueError):
        next(delu.iter_batches({}, 1))
    with pytest.raises(ValueError):
        next(delu.iter_batches(Empty(), 1))

    # Different lengths.
    a = torch.tensor([0, 1])
    b = torch.tensor([0, 1, 2])
    with pytest.raises(ValueError):
        next(delu.iter_batches((a, b), 2))
    with pytest.raises(ValueError):
        next(delu.iter_batches({'a': a, 'b': b}, 2))
    with pytest.raises(ValueError):
        next(delu.iter_batches(PointDC(a, b), 2))
