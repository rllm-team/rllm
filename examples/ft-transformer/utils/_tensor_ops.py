import dataclasses
import typing
from typing import Any, Callable, Iterable, Iterator, List, Optional, TypeVar

import torch
import torch.nn as nn

from ._utils import deprecated

T = TypeVar('T')
K = TypeVar('K')


def to(obj: T, /, *args, **kwargs) -> T:
    """Change devices and data types of tensors and modules in an arbitrary Python object.

    The two primary use cases for this function are changing the device and data types
    of tensors and modules that are a part of:

    - a complex Python object (e.g. a training state, checkpoint, etc.)
    - an object of an unknown type (when implementing generic pipelines)

    .. list-table::
        :widths: 30 50
        :header-rows: 1

        * - ``type(obj)``
          - What `delu.to(obj, *args, **kwargs)` returns
        * - `bool` `int` `float` `str` `bytes`
          - ``obj`` is returned as-is.
        * - `tuple` `list` `set` `frozenset` and all other
            collections falling under
            `typing.Sequence` `typing.Set` `typing.FrozenSet`
          - A new collection of the same type where `delu.to`
            is recursively applied to all items.
        * - `dict` or any other `typing.Mapping`
          - A new collection of the same type where `delu.to`
            is recursively applied to all keys and values.
        * - `torch.Tensor`
          - ``obj.to(*args, **kwargs)``
        * - `torch.nn.Module`
          - (``obj`` **is modified in-place**)
            ``obj.to(*args, **kwargs)``
        * - Any other type (custom classes are allowed)
          - (``obj`` **is modified in-place**)
            ``obj`` itself with all attributes recursively updated with `delu.to`.

    **Usage**

    Trivial immutable objects are returned as-is:

    >>> kwargs = {'device': 'cpu', 'dtype': torch.half}
    >>>
    >>> x = 0
    >>> x_new = delu.to(x, **kwargs)
    >>> x_new is x
    True

    If a collection is passed, a new one is created.
    The behavior for the nested values depends on their types:

    >>> x = {
    ...     # The "unchanged" tensor will not be changed,
    ...     # because it already has the requested dtype and device.
    ...     'unchanged': torch.tensor(0, **kwargs),
    ...     'changed': torch.tensor(1),
    ...     'module': nn.Linear(2, 3),
    ...     'other': [False, 1, 2.0, 'hello', b'world']
    ... }
    >>> x_new = delu.to(x, **kwargs)
    >>> # The collection itself is a new object:
    >>> x_new is x
    False
    >>> # Tensors change according to `torch.Tensor.to`:
    >>> x_new['unchanged'] is x['unchanged']
    True
    >>> x_new['changed'] is x['changed']
    False
    >>> # Modules are modified in-place:
    >>> x_new['module'] is x['module']
    True

    Complex user-defined types are also allowed:

    >>> from dataclasses import dataclass
    >>>
    >>> class A:
    ...     def __init__(self):
    ...         self.a = torch.randn(5)
    ...         self.b = ('Hello, world!', torch.randn(10))
    ...         self.c = nn.Linear(4, 7)
    ...
    >>> @dataclass
    >>> class B:
    ...     d: List[A]
    ...
    >>> x = B([A(), A()])
    >>> x_new = delu.to(x, **kwargs)
    >>> # The object is the same in terms of Python `id`,
    >>> # however, some of its nested attributes changed.
    >>> x_new is x
    True

    Args:
        obj: the input object.
        args: the positional arguments for `torch.Tensor.to`/`torch.nn.Module.to`.
        kwargs: the keyword arguments for `torch.Tensor.to`/`torch.nn.Module.to`.

    Returns:
        the transformed object.
    """  # noqa: E501
    # mypy does not understand what is going on here, hence a lot of "type: ignore"

    if isinstance(obj, (torch.Tensor, nn.Module)):
        return obj.to(*args, **kwargs)  # type: ignore

    if obj is None or isinstance(obj, (bool, int, float, str, bytes)):
        return obj  # type: ignore

    elif isinstance(obj, (typing.Sequence, typing.Set, typing.FrozenSet)):
        constructor = type(obj)
        if issubclass(constructor, tuple):
            # Handle named tuples.
            constructor = getattr(constructor, '_make', constructor)
        return constructor(to(x, *args, **kwargs) for x in obj)  # type: ignore

    elif isinstance(obj, typing.Mapping):
        # Tensors can be keys.
        return type(obj)(
            (to(k, *args, **kwargs), to(v, *args, **kwargs)) for k, v in obj.items()
        )  # type: ignore

    else:
        for attr in obj.__slots__ if hasattr(obj, '__slots__') else obj.__dict__:
            try:
                setattr(obj, attr, to(getattr(obj, attr), *args, **kwargs))
            except Exception as err:
                raise RuntimeError(
                    f'Failed to update the attribute {attr}'
                    f' of the (perhaps, nested) value of the type {type(obj)}'
                    ' with the `delu.to` function'
                ) from err
        return obj


def cat(data: List[T], /, dim: int = 0) -> T:
    """Concatenate a sequence of collections of tensors.

    `delu.cat` is a generalized version of `torch.cat` for concatenating
    not only tensors, but also (nested) collections of tensors.

    **Usage**

    Let's see how a sequence of model outputs for batches can be concatenated
    into a output tuple for the whole dataset:

    >>> from torch.utils.data import DataLoader, TensorDataset
    >>> dataset = TensorDataset(torch.randn(320, 24))
    >>> batch_size = 32
    >>>
    >>> # The model returns not only predictions, but also embeddings.
    >>> def model(x_batch):
    ...     # A dummy forward pass.
    ...     embeddings_batch = torch.randn(batch_size, 16)
    ...     y_pred_batch = torch.randn(batch_size)
    ...     return (y_pred_batch, embeddings_batch)
    ...
    >>> y_pred, embeddings = delu.cat(
    ...     [model(batch) for batch in DataLoader(dataset, batch_size, shuffle=True)]
    ... )
    >>> len(y_pred) == len(dataset)
    True
    >>> len(embeddings) == len(dataset)
    True

    The same works for dictionaries:

    >>> def model(x_batch):
    ...     return {
    ...         'y_pred': torch.randn(batch_size),
    ...         'embeddings': torch.randn(batch_size, 16)
    ...     }
    ...
    >>> outputs = delu.cat(
    ...     [model(batch) for batch in DataLoader(dataset, batch_size, shuffle=True)]
    ... )
    >>> len(outputs['y_pred']) == len(dataset)
    True
    >>> len(outputs['embeddings']) == len(dataset)
    True

    The same works for sequences of named tuples, dataclasses, tensors and
    nested combinations of all mentioned collection types.

    *Below, additinal technical examples are provided.*

    The common setup:

    >>> # First batch.
    >>> x1 = torch.randn(64, 10)
    >>> y1 = torch.randn(64)
    >>> # Second batch.
    >>> x2 = torch.randn(64, 10)
    >>> y2 = torch.randn(64)
    >>> # The last (incomplete) batch.
    >>> x3 = torch.randn(7, 10)
    >>> y3 = torch.randn(7)
    >>> total_size = len(x1) + len(x2) + len(x3)

    `delu.cat` can be applied to tuples:

    >>> batches = [(x1, y1), (x2, y2), (x3, y3)]
    >>> X, Y = delu.cat(batches)
    >>> len(X) == total_size and len(Y) == total_size
    True

    `delu.cat` can be applied to dictionaries:

    >>> batches = [
    ...     {'x': x1, 'y': y1},
    ...     {'x': x2, 'y': y2},
    ...     {'x': x3, 'y': y3},
    ... ]
    >>> result = delu.cat(batches)
    >>> isinstance(result, dict)
    True
    >>> len(result['x']) == total_size and len(result['y']) == total_size
    True

    `delu.cat` can be applied to named tuples:

    >>> from typing import NamedTuple
    >>> class Data(NamedTuple):
    ...     x: torch.Tensor
    ...     y: torch.Tensor
    ...
    >>> batches = [Data(x1, y1), Data(x2, y2), Data(x3, y3)]
    >>> result = delu.cat(batches)
    >>> isinstance(result, Data)
    True
    >>> len(result.x) == total_size and len(result.y) == total_size
    True

    `delu.cat` can be applied to dataclasses:

    >>> from dataclasses import dataclass
    >>> @dataclass
    ... class Data:
    ...     x: torch.Tensor
    ...     y: torch.Tensor
    ...
    >>> batches = [Data(x1, y1), Data(x2, y2), Data(x3, y3)]
    >>> result = delu.cat(batches)
    >>> isinstance(result, Data)
    True
    >>> len(result.x) == total_size and len(result.y) == total_size
    True

    `delu.cat` can be applied to nested collections:

    >>> batches = [
    ...     (x1, {'a': {'b': y1}}),
    ...     (x2, {'a': {'b': y2}}),
    ...     (x3, {'a': {'b': y3}}),
    ... ]
    >>> X, Y_nested = delu.cat(batches)
    >>> len(X) == total_size and len(Y_nested['a']['b']) == total_size
    True

    **Lists are not supported:**

    >>> # This does not work. Instead, use tuples.
    >>> # batches = [[x1, y1], [x2, y2], [x3, y3]]
    >>> # delu.cat(batches)  # Error

    Args:
        data: the list of collections of tensors.
            All items of the list must be of the same type, structure and layout, only
            the ``dim`` dimension can vary (same as for `torch.cat`).
            All the "leaf" values must be of the type `torch.Tensor`.
        dim: the dimension along which the tensors are concatenated.
    Returns:
        The concatenated items of the list.
    """
    if not isinstance(data, list):
        raise ValueError('The input must be a list')
    if not data:
        raise ValueError('The input must be non-empty')

    first = data[0]

    if isinstance(first, torch.Tensor):
        return torch.cat(data, dim=dim)  # type: ignore

    elif isinstance(first, tuple):
        constructor = type(first)
        constructor = getattr(constructor, '_make', constructor)  # Handle named tuples.
        return constructor(
            cat([x[i] for x in data], dim=dim) for i in range(len(first))  # type: ignore
        )

    elif isinstance(first, dict):
        return type(first)((key, cat([x[key] for x in data], dim=dim)) for key in first)  # type: ignore

    elif dataclasses.is_dataclass(first):
        return type(first)(
            **{
                field.name: cat([getattr(x, field.name) for x in data], dim=dim)
                for field in dataclasses.fields(first)
            }
        )  # type: ignore

    else:
        raise ValueError(f'The collection type {type(first)} is not supported.')


def _make_index_batches(
    x: torch.Tensor,
    batch_size: int,
    shuffle: bool,
    generator: Optional[torch.Generator],
    drop_last: bool,
) -> Iterable[torch.Tensor]:
    size = len(x)
    if not size:
        raise ValueError('data must not contain empty tensors')
    batch_indices = (
        torch.randperm(size, generator=generator, device=x.device)
        if shuffle
        else torch.arange(size, device=x.device)
    ).split(batch_size)
    return (
        batch_indices[:-1]
        if batch_indices and drop_last and len(batch_indices[-1]) < batch_size
        else batch_indices
    )


def iter_batches(
    data: T,
    /,
    batch_size: int,
    *,
    shuffle: bool = False,
    generator: Optional[torch.Generator] = None,
    drop_last: bool = False,
) -> Iterator[T]:
    """Iterate over a tensor or a collection of tensors by (random) batches.

    The function makes batches along the first dimension of the tensors in ``data``.

    TL;DR (assuming that ``X`` and ``Y`` denote full tensors
    and ``xi`` and ``yi`` denote batches):

    - ``delu.iter_batches: X -> [x1, x2, ..., xN]``
    - ``delu.iter_batches: (X, Y) -> [(x1, y1), (x2, y2), ..., (xN, yN)]``
    - ``delu.iter_batches: {'x': X, 'y': Y} -> [{'x': x1, 'y': y1}, ...]``
    - Same for named tuples.
    - Same for dataclasses.

    .. note::
        `delu.iter_batches` is significantly faster for in-memory tensors
        than `torch.utils.data.DataLoader`, because, when building batches,
        it uses batched indexing instead of one-by-one indexing.

    **Usage**

    >>> X = torch.randn(12, 32)
    >>> Y = torch.randn(12)

    `delu.iter_batches` can be applied to tensors:

    >>> for x in delu.iter_batches(X, batch_size=5):
    ...     print(len(x))
    5
    5
    2

    `delu.iter_batches` can be applied to tuples:

    >>> # shuffle=True can be useful for training.
    >>> dataset = (X, Y)
    >>> for x, y in delu.iter_batches(dataset, batch_size=5, shuffle=True):
    ...     print(len(x), len(y))
    5 5
    5 5
    2 2
    >>> # Drop the last incomplete batch.
    >>> for x, y in delu.iter_batches(
    ...     dataset, batch_size=5, shuffle=True, drop_last=True
    ... ):
    ...     print(len(x), len(y))
    5 5
    5 5
    >>> # The last batch is complete, so drop_last=True does not have any effect.
    >>> batches = []
    >>> for x, y in delu.iter_batches(dataset, batch_size=6, drop_last=True):
    ...     print(len(x), len(y))
    ...     batches.append((x, y))
    6 6
    6 6

    By default, ``shuffle`` is set to `False`, i.e. the order of items is preserved:

    >>> X2, Y2 = delu.cat(list(delu.iter_batches((X, Y), batch_size=5)))
    >>> print((X == X2).all().item(), (Y == Y2).all().item())
    True True

    `delu.iter_batches` can be applied to dictionaries:

    >>> dataset = {'x': X, 'y': Y}
    >>> for batch in delu.iter_batches(dataset, batch_size=5, shuffle=True):
    ...     print(isinstance(batch, dict), len(batch['x']), len(batch['y']))
    True 5 5
    True 5 5
    True 2 2

    `delu.iter_batches` can be applied to named tuples:

    >>> from typing import NamedTuple
    >>> class Data(NamedTuple):
    ...     x: torch.Tensor
    ...     y: torch.Tensor
    >>> dataset = Data(X, Y)
    >>> for batch in delu.iter_batches(dataset, batch_size=5, shuffle=True):
    ...     print(isinstance(batch, Data), len(batch.x), len(batch.y))
    True 5 5
    True 5 5
    True 2 2

    `delu.iter_batches` can be applied to dataclasses:

    >>> from dataclasses import dataclass
    >>> @dataclass
    ... class Data:
    ...     x: torch.Tensor
    ...     y: torch.Tensor
    >>> dataset = Data(X, Y)
    >>> for batch in delu.iter_batches(dataset, batch_size=5, shuffle=True):
    ...     print(isinstance(batch, Data), len(batch.x), len(batch.y))
    True 5 5
    True 5 5
    True 2 2

    Args:
        data: the tensor or the non-empty collection of tensors.
            If data is a collection, then the tensors must be of the same size
            along the first dimension.
        batch_size: the batch size. If ``drop_last`` is False,
            then the last batch can be smaller than ``batch_size``.
        shuffle: if True, iterate over random batches (without replacement),
            not sequentially.
        generator: when ``shuffle`` is True, passing ``generator`` makes the function
            reproducible.
        drop_last: when ``True`` and the last batch is smaller then ``batch_size``,
            then this last batch is not returned
            (in other words,
            same as the ``drop_last`` argument for `torch.utils.data.DataLoader`).
    Returns:
        the iterator over batches.
    """
    if not shuffle and generator is not None:
        raise ValueError('When shuffle is False, generator must be None.')

    constructor: Callable[[Any], T]
    args = (batch_size, shuffle, generator, drop_last)

    if isinstance(data, torch.Tensor):
        item = data
        for idx in _make_index_batches(item, *args):
            yield data[idx]  # type: ignore

    elif isinstance(data, tuple):
        if not data:
            raise ValueError('data must be non-empty')
        item = data[0]
        for x in data:
            if not isinstance(x, torch.Tensor) or len(x) != len(item):
                raise ValueError(
                    'If data is a tuple, it must contain only tensors,'
                    ' and they must have the same first dimension'
                )
        constructor = type(data)  # type: ignore
        constructor = getattr(constructor, '_make', constructor)  # Handle named tuples.
        for idx in _make_index_batches(item, *args):
            yield constructor(x[idx] for x in data)

    elif isinstance(data, dict):
        if not data:
            raise ValueError('data must be non-empty')
        item = next(iter(data.values()))
        for x in data.values():
            if not isinstance(x, torch.Tensor) or len(x) != len(item):
                raise ValueError(
                    'If data is a dict, it must contain only tensors,'
                    ' and they must have the same first dimension'
                )
        constructor = type(data)  # type: ignore
        for idx in _make_index_batches(item, *args):
            yield constructor((k, v[idx]) for k, v in data.items())

    elif dataclasses.is_dataclass(data):
        fields = list(dataclasses.fields(data))
        if not fields:
            raise ValueError('data must be non-empty')
        item = getattr(data, fields[0].name)
        for field in fields:
            if field.type is not torch.Tensor:
                raise ValueError('All dataclass fields must be tensors.')
            if len(getattr(data, field.name)) != len(item):
                raise ValueError(
                    'All dataclass tensors must have the same first dimension.'
                )
        constructor = type(data)  # type: ignore
        for idx in _make_index_batches(item, *args):
            yield constructor(
                **{field.name: getattr(data, field.name)[idx] for field in fields}  # type: ignore
            )

    else:
        raise ValueError(f'The collection {type(data)} is not supported.')


@deprecated('Instead, use `delu.cat`.')
def concat(*args, **kwargs):
    """
    <DEPRECATION MESSAGE>
    """
    return cat(*args, **kwargs)
