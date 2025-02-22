import copy
import math
import warnings
from abc import ABC
from typing import Any, Dict, List, Literal, Optional, Union
from dataclasses import dataclass

import torch
import numpy as np
import pandas as pd
from torch import Tensor

from rllm.rf.relationframe import Relation


class BaseSampler(ABC):
    r"""
    BaseSampler is the base class for all samplers.
    """

    def sample(self, *args, **kwargs) -> Any:
        raise NotImplementedError


# class SamplerOutput:
#     r"""
#     SamplerOutput is the output of a sampler.
#     """

#     def __init__(self, **kwargs):
#         self.__dict__.update(kwargs)

#     def __repr__(self):
#         return str(self.__dict__)
    

# @dataclass
# class Block:
#     """
#     Edge list
#     src nodes --(rel)--> dst nodes
#     """
#     src: List[int] 
#     dst: List[int]
#     rel: Relation

@dataclass
class Block:
    # TODO: Arbitrary edge attributes
    edge_list : Union[np.ndarray, Tensor]
    # edge_val = Tensor
    rel: Relation

    def __repr__(self):
        return f"Block( edge_list: ({self.edge_list.shape}), rel: {self.rel})"