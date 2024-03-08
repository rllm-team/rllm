import inspect
import secrets
from contextlib import ContextDecorator
from typing import Any, Optional

import torch
import torch.nn as nn

from . import random as delu_random
from ._utils import deprecated


@deprecated(
    'Instead, use `delu.random.seed` and manually set flags mentioned'
    ' in the `PyTorch docs on reproducibility <https://pytorch.org/docs/stable/notes/randomness.html>`_'
)
def improve_reproducibility(
    base_seed: Optional[int], one_cuda_seed: bool = False
) -> int:
    """Set seeds and turn off non-deterministic algorithms.

    <DEPRECATION MESSAGE>

    Do everything possible to improve reproducibility for code that relies on global
    random number generators. See also the note below.

    Sets:

    1. seeds in `random`, `numpy.random`, `torch`, `torch.cuda`
    2. `torch.backends.cudnn.benchmark` to `False`
    3. `torch.backends.cudnn.deterministic` to `True`

    Args:
        base_seed: the argument for `delu.random.seed`. If `None`, a high-quality base
            seed is generated instead.
        one_cuda_seed: the argument for `delu.random.seed`.

    Returns:
        base_seed: if ``base_seed`` is set to `None`, the generated base seed is
            returned; otherwise, ``base_seed`` is returned as is

    Note:
        If you don't want to choose the base seed, but still want to have a chance to
        reproduce things, you can use the following pattern::

            print('Seed:', delu.improve_reproducibility(None))

    Note:
        100% reproducibility is not always possible in PyTorch. See
        `this page <https://pytorch.org/docs/stable/notes/randomness.html>`_ for
        details.

    Examples:
        .. testcode::

            assert delu.improve_reproducibility(0) == 0
            seed = delu.improve_reproducibility(None)
    """
    torch.backends.cudnn.benchmark = False  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    if base_seed is None:
        # See https://numpy.org/doc/1.18/reference/random/bit_generators/index.html#seeding-and-entropy  # noqa
        base_seed = secrets.randbits(128) % (2**32 - 1024)
    else:
        assert base_seed < (2**32 - 1024)
    delu_random.seed(base_seed, one_cuda_seed=one_cuda_seed)
    return base_seed


@deprecated('Instead, use ``model.eval()`` + ``torch.no_inference/no_grad``')
class evaluation(ContextDecorator):
    """Context-manager & decorator for models evaluation.

    <DEPRECATION MESSAGE>

    This code... ::

        with delu.evaluation(model):  # or: evaluation(model_0, model_1, ...)
            ...
        @delu.evaluation(model)  # or: @evaluation(model_0, model_1, ...)
        def f():
            ...

    ...is equivalent to the following ::

        context = getattr(torch, 'inference_mode', torch.no_grad)
        with context():
            model.eval()
            ...
        @context()
        def f():
            model.eval()
            ...

    Args:
        modules
    Note:
        The training status of modules is undefined once a context is finished or a
        decorated function returns.
    Warning:
        The function must be used in the same way as `torch.no_grad` and
        `torch.inference_mode`, i.e. only as a context manager or a decorator as shown
        below in the examples. Otherwise, the behaviour is undefined.
    Warning:
        Contrary to `torch.no_grad` and `torch.inference_mode`, the function cannot be
        used to decorate generators. So, in the case of generators, you have to manually
        create a context::

            def my_generator():
                with delu.evaluation(...):
                    for a in b:
                        yield c

    Examples:

        .. testcode::

            a = torch.nn.Linear(1, 1)
            b = torch.nn.Linear(2, 2)
            with delu.evaluation(a):
                ...
            with delu.evaluation(a, b):
                ...
            @delu.evaluation(a)
            def f():
                ...
            @delu.evaluation(a, b)
            def f():
                ...
    """

    def __init__(self, *modules: nn.Module) -> None:
        assert modules
        self._modules = modules
        self._torch_context: Any = None

    def __call__(self, func):
        """Decorate a function with an evaluation context.

        Args:
            func
        """
        assert not inspect.isgeneratorfunction(
            func
        ), f'{self.__class__} cannot be used to decorate generators.'
        return super().__call__(func)

    def __enter__(self) -> None:
        assert self._torch_context is None
        self._torch_context = getattr(torch, 'inference_mode', torch.no_grad)()
        self._torch_context.__enter__()  # type: ignore
        for m in self._modules:
            m.eval()

    def __exit__(self, *exc):
        assert self._torch_context is not None
        result = self._torch_context.__exit__(*exc)  # type: ignore
        self._torch_context = None
        return result
