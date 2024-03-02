import random

import numpy as np
import torch
from pytest import raises

import delu

from .util import requires_gpu


def _test_seed(functions):
    high = 1000000
    for seed in range(10):
        x = [None, None]
        for i in range(2):
            delu.random.seed(seed)
            x[i] = [f(high) for f in functions]
        assert x[0] == x[1]


def test_seed_cpu():
    _test_seed(
        [
            lambda x: random.randint(0, x),
            lambda x: np.random.randint(x),
            lambda x: torch.randint(x, (1,))[0].item(),
        ]
    )
    with raises(ValueError):
        delu.random.seed(-1)


@requires_gpu
def test_seed_gpu():
    functions = []
    for i in range(torch.cuda.device_count()):

        def f(x):
            return (torch.randint(x, (1,), device=f'cuda:{i}')[0].item(),)

        functions.append(f)
    _test_seed(functions)


def test_get_set_state():
    high = 1000000

    def f():
        size = (1,)
        x = (
            random.randint(0, high),
            np.random.randint(0, high, size),
            torch.randint(0, high, size),
        )
        if torch.cuda.is_available():
            x += tuple(
                torch.randint(0, high, size, device=f'cuda:{d}')
                for d in range(torch.cuda.device_count())
            )
        return x

    state = delu.random.get_state()
    value = f()
    for _ in range(10):
        delu.random.set_state(state)
        assert value == f()

    if torch.cuda.is_available():
        state['torch.cuda'] = []
        with raises(AssertionError):
            delu.random.set_state(state)
    else:
        state['torch.cuda'] = [None]
        with raises(RuntimeError):
            delu.random.set_state(state)
