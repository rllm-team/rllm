from abc import ABC
from typing import Any, Union
from dataclasses import dataclass

import numpy as np
import torch
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
    """
    Edge direction: fkey_table.fkey ----> pkey_table.pkey
    edge_list: (src, dst), i.e., (fkey, pkey)
    """
    edge_list: Union[np.ndarray, Tensor]
    # edge_val = Tensor
    rel: Relation
    src_nodes: Union[np.ndarray, Tensor]
    dst_nodes: Union[np.ndarray, Tensor]

    @property
    def src_nodes(self):
        return self.src_nodes

    @src_nodes.setter
    def src_nodes(self, src_nodes: Union[np.ndarray, Tensor]):
        if isinstance(src_nodes, np.ndarray):
            self.src_nodes = np.unique(src_nodes)
        elif isinstance(src_nodes, Tensor):
            self.src_nodes = torch.unique(src_nodes)

    @property
    def dst_nodes(self):
        return self.dst_nodes

    @dst_nodes.setter
    def dst_nodes(self, dst_nodes: Union[np.ndarray, Tensor]):
        if isinstance(dst_nodes, np.ndarray):
            self.dst_nodes = np.unique(dst_nodes)
        elif isinstance(dst_nodes, Tensor):
            self.dst_nodes = torch.unique(dst_nodes)

    def __repr__(self):
        return f"Block( edge_list: ({self.edge_list.shape}), rel: {self.rel})"
