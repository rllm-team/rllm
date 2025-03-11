from typing import Union, Type, Iterable

import torch
from torch import Tensor
import torch.utils
import torch.utils.data

from rllm.data import RelationFrame, TableData
from rllm.sampler import BaseSampler, FPkeySampler
from rllm.dataloader.base import LoaderMixin


class EntryLoader(
    torch.utils.data.DataLoader,
    LoaderMixin
):
    r"""
    EntryLoader is a dataloader for the entry-level task.

    Args:
        set_ (Union[Tensor, Iterable]): the set of entry indices to sample.
        seed_table (TableData): the seed table.
        sampling (bool): whether to sample.
        rf (RelationFrame): the relation frame.
        Sampler (Type[BaseSampler]): the sampler.
        batch_size (int): the batch size.
    """

    def __init__(
        self,
        set_: Union[Tensor, Iterable],
        seed_table: TableData,
        *,
        sampling: bool,
        rf: RelationFrame,
        Sampler: Type[BaseSampler] = FPkeySampler,
        batch_size: int = 1,
        **kwargs
    ):
        self.seed_table = seed_table
        self.set_ = self.unify_set_(set_)

        if sampling:
            if Sampler is None or rf is None:
                raise ValueError("sampler and rf should not be None.")
            elif not issubclass(Sampler, BaseSampler):
                raise ValueError("sampler should be a subclass of BaseSampler.")
            self.sampling = True
            self._sampler, kwargs = self.init_sampler(Sampler, rf, seed_table, **kwargs)

        self.batch_size = batch_size

        super().__init__(
            self.set_, batch_size=batch_size, collate_fn=self.collate_fn, **kwargs
        )

    def __call__(self, index: Union[Tensor, Iterable]) -> RelationFrame:
        if isinstance(index, Tensor):
            index = index.tolist()

        index = self.set_[index]
        out = self.collate_fn(index)
        return out

    def collate_fn(self, index: Iterable):
        if self.sampling:
            return self._sampler.sample(index)
        else:
            return RelationFrame(self.seed_table)
