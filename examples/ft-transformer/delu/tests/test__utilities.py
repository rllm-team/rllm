import random

import numpy as np
import pytest
import torch

import delu

from .util import ignore_deprecated_warning


@ignore_deprecated_warning
@pytest.mark.parametrize('train', [False, True])
@pytest.mark.parametrize('grad', [False, True])
@pytest.mark.parametrize('n_models', range(3))
def test_evaluation(train, grad, n_models):
    if not n_models:
        with pytest.raises(AssertionError):
            with delu.evaluation():
                pass
        return

    torch.set_grad_enabled(grad)
    models = [torch.nn.Linear(1, 1) for _ in range(n_models)]
    for x in models:
        x.train(train)
    with delu.evaluation(*models):
        assert all(not x.training for x in models[:-1])
        assert not torch.is_grad_enabled()
    assert torch.is_grad_enabled() == grad
    for x in models:
        x.train(train)

    @delu.evaluation(*models)
    def f():
        assert all(not x.training for x in models[:-1])
        assert not torch.is_grad_enabled()
        for x in models:
            x.train(train)

    for _ in range(3):
        f()
        assert torch.is_grad_enabled() == grad


@ignore_deprecated_warning
def test_evaluation_generator():
    with pytest.raises(AssertionError):

        @delu.evaluation(torch.nn.Linear(1, 1))
        def generator():
            yield 1


@ignore_deprecated_warning
def test_improve_reproducibility():
    def f():
        upper_bound = 100
        return [
            random.randint(0, upper_bound),
            np.random.randint(upper_bound),
            torch.randint(upper_bound, (1,))[0].item(),
        ]

    for seed in [None, 0, 1, 2]:
        seed = delu.improve_reproducibility(seed)
        assert not torch.backends.cudnn.benchmark
        assert torch.backends.cudnn.deterministic
        results = f()
        delu.random.seed(seed)
        assert results == f()
