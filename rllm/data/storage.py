import copy
import weakref
from itertools import chain
from typing import Any, Dict, Optional, Callable, Mapping, Sequence, Union
from collections.abc import MutableMapping

import torch
from torch import Tensor

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

    def __init__(self, initialdata: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(initialdata, **kwargs)

    @property
    def num_nodes(self):
        if "num_nodes" in self:
            return self["num_nodes"]

        for key, value in self.items():
            if key in {"x", "pos", "batch"} or "node" in key:
                return len(value)

        return -1


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
        from rllm.utils.sparse import is_torch_sparse_tensor

        if "num_edges" in self:
            return self["num_edges"]

        if "adj" in self:
            adj = self["adj"]
            if is_torch_sparse_tensor(adj):
                return adj._nnz()

        return -1

    def is_bipartite(self):
        return self._key is not None and self._key[0] != self._key[-1]


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
