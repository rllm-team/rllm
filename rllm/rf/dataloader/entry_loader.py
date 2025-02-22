from typing import Any, Optional, List, Union

import torch
from torch import Tensor
import torch.utils
import torch.utils.data
import numpy as np

from relationframe import RelationFrame
from sampler.base import BaseSampler

class EntryLoader(torch.utils.data.DataLoader):
    r"""
    relationframe(pandas) -> tensor -> sampler -> batch -> collate_fn -> filter_fn 

    e.g.
    loader = EntryLoader(rf, user_table, batch_size=32, shuffle=True)
    """
    
    def __init__(
        self,
        rf: RelationFrame,
        # seeds: Union[Tensor, np.ndarray, List[int]], # 采样的起点
        sampler: Optional[BaseSampler] = None,
        batch_size: int = 1,
        **kwargs
    ):
        self.rf = rf
        # self.seeds = seeds
        self.sampler = sampler
        self.batch_size = batch_size

        super().__init__(rf, batch_size=batch_size, collate_fn=self.collate_fn, **kwargs)
    
    def __call__(
        self,
        index: Union[Tensor, List[int]],
    ):
        r"""Samples a subgraph from a batch of input nodes."""
        out = self.collate_fn(index)
        if not self.filter_per_worker:
            out = self.filter_fn(out)
        return out
    
    def collate_fn(self, batch: List[Any]) -> Any:
        return batch