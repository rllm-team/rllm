from typing import Union, Type, Iterable

import torch
from torch import Tensor
import torch.utils
import torch.utils.data

from rllm.data import RelationFrame, TableData
from rllm.sampler import BaseSampler, FPkeySampler


class EntryLoader(torch.utils.data.DataLoader):
    r"""
    relationframe(pandas) -> tensor -> sampler -> batch -> collate_fn -> filter_fn

    e.g.
    loader = EntryLoader(rf, user_table, batch_size=32, shuffle=True)
    """

    def __init__(
        self,
        seed_table: TableData,
        set: Union[Tensor, Iterable],
        *,
        sampling: bool,
        rf: RelationFrame,
        Sampler: Type[BaseSampler] = FPkeySampler,
        batch_size: int = 1,
        **kwargs
    ):
        self.seed_table = seed_table

        if isinstance(set, Tensor):
            if set.dtype == torch.bool:
                set = set.nonzero().flatten()
            set = set.tolist()
        self.set = set

        if sampling:
            if Sampler is None or rf is None:
                raise ValueError("sampler and rf should not be None.")
            elif not issubclass(Sampler, BaseSampler):
                raise ValueError("sampler should be a subclass of BaseSampler.")
            self.sampling = True
            self._sampler = Sampler(rf, seed_table=seed_table)

        self.batch_size = batch_size

        super().__init__(
            set, batch_size=batch_size, collate_fn=self.collate_fn, **kwargs
        )

    def __call__(self, index: Union[Tensor, Iterable]) -> RelationFrame:
        if isinstance(index, Tensor):
            index = index.tolist()

        index = self.set[index]
        out = self.collate_fn(index)
        return out

    def collate_fn(self, index: Iterable):
        if self.sampling:
            return self._sampler.sample(index)
        else:
            return RelationFrame(self.seed_table)
