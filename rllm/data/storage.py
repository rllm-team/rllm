from __future__ import annotations
import copy
import weakref
from itertools import chain
from typing import Any, Dict, Optional, Callable, Mapping, Sequence, Union, Tuple
from collections.abc import MutableMapping

import torch
from torch import Tensor
import numpy as np

from rllm.utils import is_torch_sparse_tensor, _to_csc
from rllm.data.view import KeysView, ValuesView, ItemsView


class BaseStorage(MutableMapping):
    r"""A base class for storing nodes or edges in a graph.
    This class wraps a Python dictionary and extends it as follows:

    1. It allows attribute assignments, e.g.:
       `storage.x = ...` in addition to `storage['x'] = ...`
    2. It allows private attributes that are not exposed to the user, e.g.:
       `storage._{key} = ...` and accessible via `storage._{key}`
    3. It holds an (optional) weak reference to its parent object, e.g.:
       `storage._parent = weakref.ref(parent)`
    4. It adds additional PyTorch Tensor functionality, e.g.:
       `storage.cpu()`, `storage.cuda()`.
    """

    def __init__(self, initialdata: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__()
        self._mapping = {}

        for key, value in chain((initialdata or {}).items(), kwargs.items()):
            setattr(self, key, value)

    def __len__(self):
        return len(self._mapping)

    def apply(self, func: Callable, *args: str):
        for key, value in self.items(*args):
            self[key] = recursive_apply(value, func)
        return self

    def to(self, device: Union[int, str], *args: str, non_blocking: bool = False):
        return self.apply(
            lambda x: x.to(device=device, non_blocking=non_blocking), *args
        )

    def cpu(self, *args: str):
        return self.apply(lambda x: x.cpu(), *args)

    def cuda(
        self,
        device: Optional[Union[int, str]] = None,
        *args: str,
        non_blocking: bool = False,
    ):
        return self.apply(lambda x: x.cuda(device, non_blocking=non_blocking), *args)

    def pin_memory(self, *args: str):
        return self.apply(lambda x: x.pin_memory(), *args)

    def __setattr__(self, key: str, value: Any):
        propobj = getattr(self.__class__, key, None)
        if propobj is not None and getattr(propobj, "fset", None) is not None:
            propobj.fset(self, value)
        elif key == "_parent":
            self.__dict__[key] = weakref.ref(value)
        elif key[:1] == "_":
            self.__dict__[key] = value
        else:
            self[key] = value

    def __getattr__(self, key: str):
        # avoid infinite loop.
        if key == "_mapping":
            self._mapping = {}
            return self._mapping
        try:
            return self[key]
        except KeyError:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{key}'"
            )

    def __delattr__(self, key: str) -> None:
        if key[:1] == "_":
            del self.__dict__[key]
        else:
            del self[key]

    def keys(self, *args: str):
        return KeysView(self._mapping, *args)

    def values(self, *args: str):
        return ValuesView(self._mapping, *args)

    def items(self, *args: str):
        return ItemsView(self._mapping, *args)

    def __setitem__(self, key: str, value: Any):
        self._mapping[key] = value

    def __getitem__(self, key: str):
        if key in self._mapping:
            return self._mapping[key]
        if hasattr(self.__class__, "__missing__"):
            return self.__class__.__missing__(self, key)
        raise KeyError(key)

    def __delitem__(self, key: str):
        del self._mapping[key]

    def to_dict(self) -> Dict[str, Any]:
        out_dict = copy.copy(self._mapping)
        return out_dict

    def get(self, key: str, value: Optional[Any] = None) -> Any:
        return self._mapping.get(key, value)

    def __iter__(self):
        return iter(self._mapping)

    def __contains__(self, key: str):
        return key in self._mapping

    def __copy__(self) -> 'BaseStorage':
        out = self.__class__.__new__(self.__class__)
        for k, v in self.__dict__.items():
            out.__dict__[k] = v
        out._mapping = copy.copy(self._mapping)
        return out

    def __repr__(self):
        return repr(self._mapping)


class NodeStorage(BaseStorage):
    """A storage class for node attributes in a graph.

    Args:
        initialdata (Optional[Dict[str, Any]]): Initial data to
            populate the storage.
        **kwargs: Additional keyword arguments to populate the storage.

    Attributes:
        num_nodes (int): The number of nodes in the storage.
    """
    NODE_KEYS = {"x", "pos", "batch", "n_id"}

    def __init__(self, initialdata: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(initialdata, **kwargs)

    @property
    def num_nodes(self):
        if "num_nodes" in self:
            return self["num_nodes"]

        for key, value in self.items():
            if key in self.NODE_KEYS or "node" in key:
                return len(value)

        return -1

    # Utility functions #######################################
    def is_node_attr(self, key: str) -> bool:
        r"""Node attributes should be:
        1. List, tuple, or TableData with length equal to the number of nodes.
        2. Tensor with the first dimension equal to the number of nodes.
        3. Numpy array with the first dimension equal to the number of nodes.
        """
        if '_node_attr_cache' not in self.__dict__:
            self._node_attr_cache = set()

        if key in self._node_attr_cache:
            return True

        v = self[key]
        if (isinstance(v, (list, tuple, 'TableData')) and  # avoid circular import
                len(v) == self.num_nodes):
            self._node_attr_cache.add(key)
            return True

        elif isinstance(v, Tensor) and v.size(0) == self.num_nodes:
            self._node_attr_cache.add(key)
            return True

        elif isinstance(v, np.ndarray) and v.shape[0] == self.num_nodes:
            self._node_attr_cache.add(key)
            return True

        return False


class EdgeStorage(BaseStorage):
    """A storage class for edge attributes in a graph.

    Args:
        initialdata (Optional[Dict[str, Any]]): Initial data to
            populate the storage.
        **kwargs: Additional keyword arguments to populate the storage.

    Attributes:
        num_edges (int): The number of edges in the storage.
    """

    def __init__(self, initialdata: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(initialdata, **kwargs)

    @property
    def num_edges(self):

        if "num_edges" in self:
            return self["num_edges"]

        if "adj" in self:
            adj = self["adj"]
            if is_torch_sparse_tensor(adj):
                return adj._nnz()

        if "edge_index" in self:
            assert isinstance(self.edge_index, Tensor)
            return self.edge_index.size(1)

        return -1

    # Utility functions #######################################
    def is_bipartite(self) -> bool:
        return self._key is not None and self._key[0] != self._key[-1]

    def is_edge_attr(self, key: str) -> bool:
        r"""Edge attributes should be:
        1. List, tuple, or TableData with length equal to the number of edges.
        2. Tensor with the first dimension equal to the number of edges.
        3. Numpy array with the first dimension equal to the number of edges.
        """
        if '_edge_attr_cache' not in self.__dict__:
            self._edge_attr_cache = {'edge_index', 'adj', 'num_edges'}

        if key in self._edge_attr_cache:
            return True

        v = self[key]
        if (isinstance(v, (list, tuple, 'TableData')) and  # avoid circular import
                len(v) == self.num_edges):
            self._edge_attr_cache.add(key)
            return True

        elif isinstance(v, Tensor) and v.size(0) == self.num_edges:
            self._edge_attr_cache.add(key)
            return True

        elif isinstance(v, np.ndarray) and v.shape[0] == self.num_edges:
            self._edge_attr_cache.add(key)
            return True

        return False

    def to_csc(
        self,
        device: Optional[torch.device] = None,
        num_nodes: Optional[int] = None,
        share_memory: bool = False,
        is_sorted: bool = False,
        src_node_time: Optional[Tensor] = None,
        edge_time: Optional[Union[str, Tensor]] = None,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        r"""Convert the edge storage to a CSC format.

        Args:
            device (torch.device, optional): The desired device of the
                returned tensors. If None, use the current device.
                (default: `None`)
            num_nodes (int, optional): The number of nodes.
                If None, infer from edge_index.
                (default: `None`)
            share_memory (bool, optional): If set to `True`, will share memory
                among returned tensors. This can accelerate process when using
                multiple processes.
                (default: `False`)
            is_sorted (bool, optional): If set to `True`, will not sort the
                edge index.
                (default: `False`)
            src_node_time (Tensor, optional): The source node time.
                If not None, will sort the edge index by `src_node_time`.
                (default: `None`)
            edge_time (Union[str, Tensor], optional): The edge time attribute.
                If not None, will sort the edge index by `edge_time_attr`.
                (default: `None`)

        Returns:
            Tuple[Tensor, Tensor, Optional[Tensor]]: The column indices,
                row indices, and the permutation index.
        """
        if hasattr(self, "edge_index"):
            input = self["edge_index"]
        elif hasattr(self, "adj"):
            input = self["adj"]
        else:
            raise ValueError("No edge found. Edge type should be either `adj` or `edge_index`.")

        if isinstance(edge_time, str):
            assert edge_time in self
            edge_time = self[edge_time]

        return _to_csc(
            input=input,
            device=device,
            num_nodes=num_nodes,
            share_memory=share_memory,
            is_sorted=is_sorted,
            src_node_time=src_node_time,
            edge_time=edge_time,
        )


def recursive_apply(data: Any, func: Callable) -> Any:
    if isinstance(data, Tensor):
        return func(data)
    elif isinstance(data, torch.nn.utils.rnn.PackedSequence):
        return func(data)
    elif isinstance(data, tuple) and hasattr(data, "_fields"):  # namedtuple
        return type(data)(*(recursive_apply(d, func) for d in data))
    elif isinstance(data, Sequence) and not isinstance(data, str):
        return [recursive_apply(d, func) for d in data]
    elif isinstance(data, Mapping):
        return {key: recursive_apply(data[key], func) for key in data}
    else:
        try:
            return func(data)
        except Exception:
            return data
