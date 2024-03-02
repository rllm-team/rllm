import itertools

import pytest
import torch
import torch.nn as nn

import delu

from .util import ignore_deprecated_warning


def test_lambda():
    m = delu.nn.Lambda(torch.square)
    assert torch.allclose(m(torch.tensor(3.0)), torch.tensor(9.0))

    m = delu.nn.Lambda(torch.squeeze)
    assert m(torch.zeros(2, 1, 3, 1)).shape == (2, 3)

    m = delu.nn.Lambda(torch.squeeze, dim=1)
    assert m(torch.zeros(2, 1, 3, 1)).shape == (2, 3, 1)

    # with pytest.raises(ValueError):
    with pytest.deprecated_call():
        m = delu.nn.Lambda(lambda x: torch.square(x))
    assert torch.allclose(m(torch.tensor(3.0)), torch.tensor(9.0))

    with pytest.raises(ValueError):
        delu.nn.Lambda(torch.mul, other=torch.tensor(2.0))()


@torch.no_grad()
def test_nlinear():
    m = delu.nn.NLinear(3, 4, 5, bias=False)
    assert m.bias is None

    with pytest.raises(ValueError):
        m(torch.randn(4))
    with pytest.raises(ValueError):
        m(torch.randn(33, 4))

    for b in [(), (2,), (2, 3)]:
        for n in [(4,), (4, 5)]:
            x1 = torch.randn(*b, *n, 6)
            m = delu.nn.NLinear(n, 6, 7)
            assert m.bias is not None
            x2 = m(x1)
            assert x2.shape == (*b, *n, 7)
            for i in itertools.product(*(range(d) for d in x2.shape)):
                expected = x2[i]
                actual = torch.tensor(0.0)
                for j in range(x1.shape[-1]):
                    actual += m.weight[(*i[len(b) : -1], j, i[-1])] * x1[(*i[:-1], j)]
                actual += m.bias[i[len(b) :]]
                assert torch.allclose(actual, expected)


@ignore_deprecated_warning
def test_named_sequential():
    _ = delu.nn.named_sequential()

    m = delu.nn.named_sequential(
        ('a', nn.Linear(2, 3)), ('b', delu.nn.Lambda(torch.mul, other=0.0))
    )
    assert isinstance(m, nn.Sequential)
    assert len(m) == 2
    assert set(x[0] for x in m.named_children()) == {'a', 'b'}
    assert (m(torch.randn(2)) == 0.0).all().item()
