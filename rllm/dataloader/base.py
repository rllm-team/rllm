import inspect
from abc import ABC
from typing import Iterable, Union
from collections import OrderedDict

import torch
from torch import Tensor

from rllm.sampler import BaseSampler
from rllm.data import TableData, RelationFrame


class LoaderMixin(ABC):
    r"""Mixin class for dataloader.

    Examples::

        >>> sets = (trainset, valset, testset)
        >>> train_loader, val_loader, test_loader = XXXXLoader.create(sets, ...)
        >>> for batch in train_loader:
        >>>    ...

    Examples::

        >>> set_ = torch.arange(100)  # or torch.randint(0, 2, (lenth,), dtype=torch.bool)
        >>> loader = XXXXLoader(set_, ...)
        >>> for batch in loader:
        >>>    ...
    """

    @classmethod
    def create(cls, sets: Iterable, **kwargs):
        r"""Create dataloaders for multiple sets."""
        return (cls(set_, **kwargs) for set_ in sets)

    @staticmethod
    def init_sampler(Sampler: BaseSampler, rf: RelationFrame, seed_table: TableData, **kwargs):
        r"""Initialize the sampler. Fetch the sampler's required parameters from kwargs and pop them."""
        params = list(inspect.signature(Sampler.__init__).parameters.keys())[3:]
        params.remove('kwargs')

        init_kwargs = OrderedDict()
        init_kwargs['rf'] = rf
        init_kwargs['seed_table'] = seed_table

        for param in params:
            if param not in kwargs:
                init_kwargs[param] = None
            else:
                init_kwargs[param] = kwargs[param]
                kwargs.pop(param)

        return Sampler(**init_kwargs), kwargs

    def unify_set_(self, set_: Union[Tensor, Iterable]) -> list:
        r"""Unify the set_ to python Iterable."""
        if isinstance(set_, Tensor):
            if set_.dtype == torch.bool:
                set_ = set_.nonzero().flatten()
            set_ = set_.tolist()
        else:
            set_ = list(set_)
        return set_
