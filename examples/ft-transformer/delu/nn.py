"""An extension to `torch.nn`."""

import inspect
import warnings
from collections import OrderedDict
from typing import Callable, Tuple, Union

import torch.nn
import torch.nn as nn
from torch.nn.parameter import Parameter

from ._utils import deprecated

__all__ = ['Lambda', 'NLinear', 'named_sequential']


class Lambda(torch.nn.Module):
    """A wrapper for functions from `torch` and methods of `torch.Tensor`.

    An important "feature" of this module is that it is intentionally limited:

    - Only the functions from the `torch` module and the methods of `torch.Tensor`
      are allowed.
    - The passed callable must accept a single `torch.Tensor`
      and return a single `torch.Tensor`.
    - The allowed keyword arguments must be of simple types (see the docstring).

    **Usage**

    >>> m = delu.nn.Lambda(torch.squeeze)
    >>> m(torch.randn(2, 1, 3, 1)).shape
    torch.Size([2, 3])
    >>> m = delu.nn.Lambda(torch.squeeze, dim=1)
    >>> m(torch.randn(2, 1, 3, 1)).shape
    torch.Size([2, 3, 1])
    >>> m = delu.nn.Lambda(torch.Tensor.abs_)
    >>> m(torch.tensor(-1.0))
    tensor(1.)

    Custom functions are not allowed
    (technically, they are **temporarily** allowed,
    but this functionality is deprecated and will be removed in future releases):

    >>> # xdoctest: +SKIP
    >>> m = delu.nn.Lambda(lambda x: torch.abs(x))
    Traceback (most recent call last):
        ...
    ValueError: fn must be a function from `torch` or a method of `torch.Tensor`, but ...

    Non-trivial keyword arguments are not allowed:

    >>> m = delu.nn.Lambda(torch.mul, other=torch.tensor(2.0))
    Traceback (most recent call last):
        ...
    ValueError: For kwargs, the allowed value types include: ...
    """  # noqa: E501

    def __init__(self, fn: Callable[..., torch.Tensor], /, **kwargs) -> None:
        """
        Args:
            fn: the callable.
            kwargs: the keyword arguments for ``fn``. The allowed values types include:
                None, bool, int, float, bytes, str
                and (nested) tuples of these simple types.
        """
        super().__init__()
        if not callable(fn) or (
            fn not in vars(torch).values()
            and (
                fn not in (member for _, member in inspect.getmembers(torch.Tensor))
                or inspect.ismethod(fn)  # Check if fn is a @classmethod
            )
        ):
            warnings.warn(
                'Passing custom functions to delu.nn.Lambda is deprecated'
                ' and will be removed in future releases.'
                ' Only functions from the `torch` module and methods of `torch.Tensor`'
                ' are allowed',
                DeprecationWarning,
            )
            # NOTE: in future releases, replace the above warning with this exception:
            # raise ValueError(
            #     'fn must be a function from `torch` or a method of `torch.Tensor`,'
            #     f' but this is not true for the passed {fn=}'
            # )

        def is_valid_value(x):
            return (
                x is None
                or isinstance(x, (bool, int, float, bytes, str))
                or isinstance(x, tuple)
                and all(map(is_valid_value, x))
            )

        for k, v in kwargs.items():
            if not is_valid_value(v):
                raise ValueError(
                    'For kwargs, the allowed value types include:'
                    ' None, bool, int, float, bytes, str and (nested) tuples containing'
                    ' values of these simple types. This is not true for the passed'
                    f' argument {k} with the value {v}'
                )

        self._function = fn
        self._function_kwargs = kwargs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Do the forward pass."""
        return self._function(x, **self._function_kwargs)


class NLinear(nn.Module):
    """N *separate* linear layers for N embeddings: ``(*, *N, D1) -> (*, *N, D2)``.

    Usage examples covered below:

    - (NLP) Training a *separate* linear layer for each token embedding in a sequence.
      By contrast, using `torch.nn.Linear` would mean applying the same linear layer
      to all tokens.
    - (CV) Training a *separate* linear layer for each patch embedding in an image.
      By contrast, using `torch.nn.Linear` would mean applying the same linear layer
      to all tokens.

    Technically, ``NLinear(N, D1, D2)`` is just a layout of ``N``
    linear layers ``torch.nn.Linear(D1, D2)``.

    **Shape**

    - Input: ``(*, *n, in_features)``, where ``*`` are batch dimensions.
    - Output: ``(*, *n, out_features)``.

    **Usage**

    (NLP)
    Training a separate linear layer for each of the token embeddings in a sequence:

    >>> batch_size = 2
    >>> sequence_length = 4
    >>> d_embedding_in = 6
    >>> d_embedding_out = 7
    >>> x = torch.randn(batch_size, sequence_length, d_embedding_in)
    >>> x.shape
    torch.Size([2, 4, 6])
    >>> m = NLinear(sequence_length, d_embedding_in, d_embedding_out)
    >>> m(x).shape
    torch.Size([2, 4, 7])

    (CV)
    Training a separate linear layer for each of the patch embeddings in an image:

    >>> # Batch dimensions can also be arbitrarily complex.
    >>> batch_size = (2, 3)
    >>> width = 4
    >>> height = 5
    >>> in_channels = 6
    >>> out_channels = 7
    >>> x = torch.randn(*batch_size, width, height, in_channels)
    >>> x.shape
    torch.Size([2, 3, 4, 5, 6])
    >>> # N == width * heght == 4 * 5 == 20
    >>> m = NLinear((width, height), in_channels, out_channels)
    >>> m(x).shape
    torch.Size([2, 3, 4, 5, 7])
    """

    def __init__(
        self,
        n: Union[int, Tuple[int, ...]],
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        """
        All arguments are the same as in `torch.nn.Linear` except for ``n``,
        which is the expected layout of the input (see the examples in `NLinear`).
        """
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        n_tuple = (n,) if isinstance(n, int) else n
        if not n_tuple or any(x <= 0 for x in n_tuple):
            raise ValueError(
                'n must be a positive integer or a non-empty tuple'
                f' of positive integers. The provided value: {n=}'
            )
        self.weight = Parameter(
            torch.empty(*n_tuple, in_features, out_features, **factory_kwargs)
        )
        self.bias = (
            nn.parameter.Parameter(
                torch.empty(*n_tuple, out_features, **factory_kwargs)
            )
            if bias
            else None
        )
        self.reset_parameters()

    def reset_parameters(self):
        """Reset all parameters."""
        # The same as in torch.nn.Linear.
        d_in_rsqrt = self.weight.shape[-2] ** -0.5
        nn.init.uniform_(self.weight, -d_in_rsqrt, d_in_rsqrt)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -d_in_rsqrt, d_in_rsqrt)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Do the forward pass."""
        if x.ndim < self.weight.ndim - 1:
            raise ValueError(
                f'The input must have at least {self.weight.ndim - 1} dimentions,'
                f' but {x.ndim=}'
            )
        # The non-batch dimensions corresponding to n and in_features must be
        # exactly equal, it would be incorrect to rely on broadcasting over them.
        if x.shape[-(self.weight.ndim - 1) :] != self.weight.shape[:-1]:
            raise ValueError(
                'The input must have a shape like'
                ' `(*batch_dimensions, *n, in_features)`, where n and in_features '
                f'are the values passed to the constructor of {type(self).__name__}.'
                f' However: {x.shape=}, n={self.weight.shape[:-2]},'
                f' in_features={self.weight.shape[-2]}'
            )

        x = (x[..., None, :] @ self.weight).squeeze(-2)
        if self.bias is not None:
            x = x + self.bias
        return x


@deprecated('')
def named_sequential(*names_and_modules: Tuple[str, nn.Module]) -> nn.Sequential:
    """A shortcut for creating `torch.nn.Sequential` with named modules without using `collections.OrderedDict`.

    <DEPRECATION MESSAGE>

    The sole purpose of this function is to improve the ergonomics and readability
    of the common construction.

    **Usage**

    This ...

    >>> # xdoctest: +SKIP
    >>> m = delu.nn.named_sequential(
    ...     ('linear1', nn.Linear(10, 20)),
    ...     ('activation', nn.ReLU()),
    ...     ('linear2', nn.Linear(20, 1)),
    ... )

    ... is equivalent to this:

    >>> # xdoctest: +SKIP
    >>> from collections import OrderedDict
    >>> m = torch.nn.Sequential(
    ...     OrderedDict(
    ...         [
    ...             ('linear1', nn.Linear(10, 20)),
    ...             ('activation', nn.ReLU()),
    ...             ('linear2', nn.Linear(20, 1)),
    ...         ]
    ...     )
    ... )

    Args:
        names_and_modules: the names and the modules.
    """  # noqa: E501
    return nn.Sequential(OrderedDict(names_and_modules))
