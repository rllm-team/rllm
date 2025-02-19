from typing import Any, Optional, List, Union

import torch
from torch import Tensor
import torch.utils
import torch.utils.data
import numpy as np

from relationframe import RelationFrame

class EntryLoader(torch.utils.data.DataLoader):
    
    def __init__(
        self,
        rf: RelationFrame,
        # TODO: do i really need to use tensor here?
        seeds: Union[Tensor, np.ndarray, List[int]],
        batch_size: int = 1,
        **kwargs
    ):
        self.rf = rf
        self.seeds = seeds
        self.batch_size = batch_size

        super().__init__(rf, batch_size=batch_size, collate_fn=self.collate_fn, **kwargs)
    

    def collate_fn(self, batch: List[Any]) -> Any:
        return batch