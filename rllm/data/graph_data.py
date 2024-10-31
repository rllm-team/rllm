import copy
from typing import Any, List, Optional, Union, Callable, Mapping, Tuple
from itertools import chain

import torch
from torch import Tensor

from rllm.data.storage import BaseStorage, NodeStorage, EdgeStorage


class BaseGraph:
    """An abstract base class for graph data storage."""

    def __getattr__(self, key: str):
        raise NotImplementedError

    def __setattr__(self, key: str, value: Any):
        raise NotImplementedError

    def __delattr__(self, key: str):
        raise NotImplementedError

    # To implement `load`, `save` and `to_dict`
    # will help us define how to save this model.
    @classmethod
    def load(cls, path: str):
        raise NotImplementedError

    def save(self, path: str):
        torch.save(self.to_dict(), path)

    def to_dict(self):
        raise NotImplementedError

    def apply(self, func: Callable, *args: str):
        raise NotImplementedError

    @property
    def stores(self):
        raise NotImplementedError

    def clone(self, *args: str):
        r"""Performs cloning of tensors for the ones given in `*args`"""
        return copy.copy(self).apply(lambda x: x.clone(), *args)

    def to(self, device: Union[int, str], *args: str, non_blocking: bool = False):
        r"""Performs device conversion of the whole dataset."""
        return self.apply(
            lambda x: x.to(device=device, non_blocking=non_blocking), *args
        )

    def cpu(self, *args: str):
        r"""Moves the dataset to CPU memory."""
        return self.apply(lambda x: x.cpu(), *args)

    def cuda(
        self,
        device: Optional[Union[int, str]] = None,
        *args: str,
        non_blocking: bool = False
    ):
        r"""Moves the dataset toto CUDA memory."""
        device = "cuda" if device is None else device
        return self.apply(lambda x: x.cuda(device, non_blocking=non_blocking), *args)

    def pin_memory(self, *args: str):
        return self.apply(lambda x: x.pin_memory(), *args)

    def keys(self):
        r"""Returns a list of all graph attribute names."""
        out = []
        for store in self.stores:
            out += list(store.keys())
        return list(set(out))

    def __len__(self):
        r"""Returns the number of graph attributes."""
        return len(self.keys())

    def __contains__(self, key: str):
        r"""Returns `True` if the attribute `key` is present in the
        data.
        """
        return key in self.keys()


class GraphData(BaseGraph):
    """A class for homogenerous graph data storage which easily fit
    into CPU memory.

    Args:
        x (Tensor): Node feature matrix.
        y (Tensor): Node label matrix.
        **kwargs (optional): Additional attributes.

    Shapes:
        x: (num_nodes, num_node_features)
        y: (num_nodes,)
    """

    def __init__(
        self,
        x: Optional[Tensor] = None,
        y: Optional[Tensor] = None,
        adj: Optional[torch.sparse.FloatTensor] = None,
        **kwargs
    ):
        self._mapping = BaseStorage()

        if x is not None:
            self.x = x
        if y is not None:
            self.y = y
        if adj is not None:
            self.adj = adj

        for key, value in kwargs.items():
            setattr(self, key, value)

    # To implement `load`, `save` and `to_dict`
    # will help us define how to save this model.
    @classmethod
    def load(cls, path: str):
        data = torch.load(path)
        return cls(**data)

    def to_dict(self):
        return self._mapping.to_dict()

    def __getattr__(self, key: str):
        # avoid infinite loop.
        if key == "_mapping":
            self.__dict__["_mapping"] = BaseStorage()
            return self.__dict__["_mapping"]

        return getattr(self._mapping, key)

    def __setattr__(self, key: str, value: Any):
        propobj = getattr(self.__class__, key, None)
        if propobj is not None and getattr(propobj, "fset", None) is not None:
            propobj.fset(self, value)
        elif key[:1] == "_":
            self.__dict__[key] = value
        else:
            setattr(self._mapping, key, value)

    def __delattr__(self, key: str):
        if key[:1] == "_":
            del self.__dict__[key]
        else:
            del self._mapping[key]

    def apply(self, func: Callable, *args: str):
        self._mapping.apply(func, *args)
        return self

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

    def __iter__(self):
        return iter(self._mapping)

    @property
    def num_nodes(self):
        if "num_nodes" in self._mapping:
            return self._mapping["num_nodes"]
        return len(self.y)

    @property
    def num_classes(self):
        if "num_classes" in self._mapping:
            return self._mapping["num_classes"]
        self._mapping["num_classes"] = int(self.y.max() + 1)
        return self._mapping["num_classes"]

    @property
    def stores(self):
        r"""Returns a list of all storages of the graph."""
        return [self._mapping]

    def __len__(self):
        return len(self.x)

    def to_hetero(
        self,
        node_type: Optional[Tensor] = None,
        edge_type: Optional[Tensor] = None,
        node_type_names: Optional[List[str]] = None,
        edge_type_names: Optional[Tuple] = None,
    ):
        r"""Converts a `GraphData` to a `HeteroGraphData`.
        Node and edge attributes are splitted as the
        `node_type` and `edge_type` vectors.

        Args:
            node_type (torch.Tensor, optional): A node-level vector denoting
                the type of each node.
            edge_type (torch.Tensor, optional): An edge-level vector denoting
                the type of each edge.
            node_type_names (List[str], optional): The names of node types.
            edge_type_names (List[Tuple], optional): The names of edge types.
        """

        from rllm.utils.sparse import get_indices

        if node_type_names is None:
            node_type_names = [str(i) for i in node_type.unique().tolist()]

        if edge_type_names is None:
            edge_type_names = []
            edges = get_indices(self.adj)
            for i in edge_type.unique().tolist():
                src, tgt = edges[:, edge_type == i]
                src_types = node_type[src].unique().tolist()
                tgt_types = node_type[tgt].unique().tolist()
                if len(src_types) != 1 and len(tgt_types) != 1:
                    raise ValueError(
                        "Could not construct a `HeteroGraphData` "
                        "object from the `GraphData` object "
                        "because single edge types span over "
                        "multiple node types"
                    )
                edge_type_names.append(
                    (
                        node_type_names[src_types[0]],
                        str(i),
                        node_type_names[tgt_types[0]],
                    )
                )

        # `index_map`` will be used when reorder edge_index
        node_ids, index_map = {}, torch.empty_like(node_type)
        for i in range(len(node_type_names)):
            node_ids[i] = (node_type == i).nonzero(as_tuple=False).view(-1)
            index_map[node_ids[i]] = torch.arange(
                len(node_ids[i]), device=index_map.device
            )

        edge_ids = {}
        for i in range(len(edge_type_names)):
            edge_ids[i] = (edge_type == i).nonzero(as_tuple=False).view(-1)

        hetero_data = HeteroGraphData()
        for i, key in enumerate(node_type_names):
            hetero_data[key].x = self.x[node_ids[i]]

        edges = get_indices(self.adj)
        # TODO: Type of adj matrix only supports sparse matrix
        values = self.adj.coalesce().values()
        for i, key in enumerate(edge_type_names):
            src, tgt = key[0], key[-1]

            num_src = hetero_data[src].num_nodes
            num_tgt = hetero_data[tgt].num_nodes

            new_edges = edges[:, edge_ids[i]]
            new_edges[0] = index_map[new_edges[0]]
            new_edges[1] = index_map[new_edges[1]]
            new_adj = torch.sparse_coo_tensor(
                new_edges, values[edge_ids[i]], (num_src, num_tgt)
            )
            hetero_data[key].adj = new_adj

        exclude_keys = set(hetero_data.keys()) | {"x", "y", "adj"}
        for attr, value in self.items():
            if attr in exclude_keys:
                continue
            setattr(hetero_data, attr, value)

        return hetero_data


class HeteroGraphData(BaseGraph):
    r"""A class for heterogenerous graph data storage which easily fit
    into CPU memory.

    Methods of initialization:
        1) Assign attributes,
        data = HeteroGraphData()
        data['paper']['x'] = x_paper
        data['paper'].x = x_paper

        2) pass them as keyword arguments,
        data = HeteroGraphData('paper' = {'x': x_paper, 'y': labels},
                            'writer' = {'x': x_writer},
                            'writer__of__paper' = {'adj' = adj})

        3) pass them as dictionaries.
        data = HeteroGraphData(
            {
                'paper' = {'x': x_paper, 'y': labels},
                'writer' = {'x': x_writer},
                ('writer', 'of', 'paper') = {'adj' = adj}
            }
        )

    Save some attributes like train_mask:
    data.train_mask = train_mask

    Save more edges and nodes:
    data[edge_type|node_type] = {
        ...
    }

    Key of edge type:
    data['src__tgt'] =  {'adj': adj}
    data[src, tgt] = {'adj': adj}
    data[src, rel, tgt] = {'adj': adj}

    Key of node type:
    data['node type'] = {'x': x}
    """

    def __init__(self, mapping: Optional[Mapping[str, Any]] = None, **kwargs):
        self._mapping = BaseStorage()
        self._nodes = {}
        self._edges = {}

        for key, value in chain((mapping or {}).items(), kwargs.items()):
            if isinstance(value, Mapping):
                self[key].update(value)
            else:
                setattr(self, key, value)

    @classmethod
    def load(cls, path: str):
        mapping = torch.load(path)
        out = cls()
        for key, value in mapping.items():
            if key == "_mapping":
                # load global variables
                out.__dict__["_mapping"] = BaseStorage(value)
            else:
                # load nodes and edges
                out[key] = value
        return out

    def to_dict(self):
        out_dict = {}
        out_dict["_mapping"] = self._mapping.to_dict()
        for key, store in chain(self._nodes.items(), self._edges.items()):
            out_dict[key] = store.to_dict()
        return out_dict

    def __getattr__(self, key: str):
        # avoid infinite loop.
        if key == "_mapping":
            self.__dict__["_mapping"] = BaseStorage()
            return self.__dict__["_mapping"]

        return getattr(self._mapping, key)

    def __setattr__(self, key: str, value: Any):
        propobj = getattr(self.__class__, key, None)
        if propobj is not None and getattr(propobj, "fset", None) is not None:
            propobj.fset(self, value)
        elif key[:1] == "_":
            self.__dict__[key] = value
        else:
            setattr(self._mapping, key, value)

    def __delattr__(self, key: str):
        if key[:1] == "_":
            del self.__dict__[key]
        else:
            del self._mapping[key]

    def apply(self, func: Callable, *args: str):
        self._mapping.apply(func, *args)
        for key, value in chain(self._nodes.items(), self._edges.items()):
            value.apply(func, *args)
        return self

    def __getitem__(self, key: Union[str, Tuple]):
        key = self._to_regulate(key)

        if isinstance(key, tuple):
            out = self._edges.get(key, None)
            if out is None:
                self._edges[key] = EdgeStorage(_parent=self, _key=key)
                out = self._edges[key]
        else:
            out = self._nodes.get(key, None)
            if out is None:
                self._nodes[key] = NodeStorage(_parent=self, _key=key)
                out = self._nodes[key]
        return out

    def __setitem__(self, key: Union[str, Tuple], value: Mapping[str, Any]):
        key = self._to_regulate(key)

        if isinstance(key, tuple):
            self._edges[key] = EdgeStorage(value)
        else:
            self._nodes[key] = NodeStorage(value)

    def __delitem__(self, key: Union[str, Tuple]):
        key = self._to_regulate(key)

        if key in self._edges:
            del self._edges[key]
        elif key in self._nodes:
            del self._nodes[key]
        elif key in self._mapping:
            del self._mapping[key]

    def _to_regulate(self, key: Union[str, Tuple]):
        if isinstance(key, str) and "__" in key:
            key = tuple(key.split("__"))
        return key

    @property
    def num_nodes(self):
        if "num_nodes" in self._mapping:
            return self._mapping["num_nodes"]

        self._mapping["num_nodes"] = sum(x.num_nodes for x in self._nodes.values())
        return self._mapping["num_nodes"]

    @property
    def node_types(self):
        r"""Returns a list of all node types of the graph."""
        return list(self._nodes.keys())

    @property
    def edge_types(self):
        r"""Returns a list of all edge types of the graph."""
        return list(self._edges.keys())

    @property
    def node_stores(self):
        r"""Returns a list of all node storages of the graph."""
        return list(self._nodes.values())

    @property
    def edge_stores(self):
        r"""Returns a list of all edge storages of the graph."""
        return list(self._edges.values())

    @property
    def stores(self):
        r"""Returns a list of all storages of the graph."""
        return [self._mapping] + self.node_stores + self.edge_stores

    def node_items(self):
        r"""Returns a list of node type and node storage pairs."""
        return list(self._nodes.items())

    def edge_items(self):
        r"""Returns a list of edge type and edge storage pairs."""
        return list(self._edges.items())

    def x_dict(self):
        r"""Collects the attribute x from all node types."""

        return {
            node_type: store.x
            for node_type, store in self._nodes.items()
            if hasattr(store, "x")
        }

    def adj_dict(self):
        r"""Collects the attribute adj from all edge types."""
        return {
            edge_type: store.adj
            for edge_type, store in self._edges.items()
            if hasattr(store, "adj")
        }

    def metadata(self):
        r"""Returns the heterogeneous meta-data, *i.e.* its node and edge
        types.

        .. code-block:: python

            data = HeteroData()
            data['paper'].x = ...
            data['author'].x = ...
            data['author', 'writes', 'paper'].edge_index = ...

            print(data.metadata())
            >>> (['paper', 'author'], [('author', 'writes', 'paper')])
        """
        return self.node_types, self.edge_types
