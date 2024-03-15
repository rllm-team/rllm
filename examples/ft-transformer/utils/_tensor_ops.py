import dataclasses
from typing import Any, Callable, Iterable, Iterator, Optional, TypeVar
import torch


T = TypeVar('T')
K = TypeVar('K')


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
        # Handle named tuples.
        constructor = getattr(constructor, '_make', constructor)
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
                # type: ignore
                **{fi.name: getattr(data, fi.name)[idx] for fi in fields}
            )

    else:
        raise ValueError(f'The collection {type(data)} is not supported.')
