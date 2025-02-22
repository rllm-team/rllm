from abc import ABC
from typing import Any, Union
from dataclasses import dataclass

import numpy as np
from torch import Tensor

from rllm.rf.relationframe import Relation


class BaseSampler(ABC):
    r"""
    BaseSampler is the base class for all samplers.
    """

    def sample(self, *args, **kwargs) -> Any:
        raise NotImplementedError


@dataclass
class Block:
    # TODO: Arbitrary edge attributes
    edge_list: Union[np.ndarray, Tensor]
    # edge_val = Tensor
    rel: Relation

    def __repr__(self):
        return f"Block( edge_list: ({self.edge_list.shape}), rel: {self.rel})"
