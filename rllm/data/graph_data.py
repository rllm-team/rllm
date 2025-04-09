import copy
from typing import Any, List, Optional, Union, Callable, Mapping, Tuple, Dict
from itertools import chain
from warnings import warn

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
        self.device = f"cuda:{device}" if isinstance(device, int) else device
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
        non_blocking: bool = False,
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
        x (Tensor, optional): Node feature matrix.
        y (Tensor, optional): Node label matrix.
        adj (torch.sparse.FloatTensor, optional): Adjacency matrix.
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
        **kwargs,
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
        data = torch.load(path, weights_only=False)
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

"""
`EdgeType` and `NodeType` are used as the types of
edges and nodes in the `HeteroGraphData` class.

`UEdgeType` is unified edge type used in `HeteroGraphData` class.
"""
EdgeType = Union[Tuple[str, str, str], str]
UEdgeType = Tuple[str, str, str]
NodeType = str


class HeteroGraphData(BaseGraph):
    r"""A class for heterogenerous graph data storage which easily fit
    into CPU memory.

    Acceptable edge key words are `adj` and `edge_index`. Other edge
    key words are considered as edge attributes.

    Methods of initialization:
        1) Assign attributes,

        data = HeteroGraphData()
        data['paper']['x'] = x_paper
        data['paper'].x = x_paper

        Tips:
            Though name of node attribute can be arbitrary, `x` is prefered.

        2) pass them as keyword arguments,

        data = HeteroGraphData(
            'paper' = {'x': x_paper, 'y': labels},
            'writer' = {'x': x_writer},
            'writer__of__paper' = {'adj' = adj}
        )

        3) pass them as dictionaries,

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
        mapping = torch.load(path, weights_only=False)
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

    # Utility function ###################################################

    def validate(self) -> bool:
        r"""Validates the graph data by checking the following:
        1. Node and edge types are matched.
        2. Edge types are valid.
        3. Edge indices are valid.
        """
        status = True

        # check dangling nodes
        node_types = set(self.node_types)
        src_n_types = {src for src, _, _ in self.edge_types}
        dst_n_types = {dst for _, _, dst in self.edge_types}
        dangling_n_types = (src_n_types | dst_n_types) - node_types
        if len(dangling_n_types) > 0:
            status = False
            warn(
                f"The node types {dangling_n_types} are referenced in edge "
                f"types, but do not exist as node types."
            )
        dangling_n_types = node_types - (src_n_types | dst_n_types)
        if len(dangling_n_types) > 0:
            warn(
                f"The node types {dangling_n_types} are isolated, "
                f"i.e. are not referenced by any edge type."
            )

        # check edges
        for edge_type, edge_store in self._edges.items():
            src, _, dst = edge_type
            n_src_nodes = self[src].num_nodes
            n_dst_nodes = self[dst].num_nodes

            if n_src_nodes is None:
                status = False
                warn(f"`num_nodes` is undefined in node type `{src}`.")

            if n_dst_nodes is None:
                status = False
                warn(f"`num_nodes` is undefined in node type `{src}`.")

            if "edge_index" in edge_store:
                edge_index = edge_store.edge_index
                if edge_index.dim() != 2 or edge_index.size(0) != 2:
                    status = False
                    warn(
                        f"`edge_index` of edge type {edge_type} needs "
                        f"to be shape [2, ...], "
                        f"but found {edge_index.size()}."
                    )

                if edge_index.numel() > 0:
                    if edge_index.min() < 0:
                        status = False
                        warn(
                            f"`edge_index` of edge type {edge_type} needs "
                            f"to be positive, "
                            f"but found {int(edge_index.min())}."
                        )
                    if edge_index[0].max() >= n_src_nodes:
                        status = False
                        warn(
                            f"src `edge_index` of edge type {edge_type} "
                            f"needs to be in range of number of src "
                            f"nodes {n_src_nodes}, "
                            f"but found {int(edge_index[0].max())}."
                        )
                    if edge_index[1].max() >= n_dst_nodes:
                        status = False
                        warn(
                            f"dst `edge_index` of edge type {edge_type} "
                            f"needs to be in range of number of dst "
                            f"nodes {n_dst_nodes}, "
                            f"but found {int(edge_index[1].max())}."
                        )
        return status

    def collect_attr(
        self,
        key: Union[str, NodeType, EdgeType],
        exlude_None: bool = False,
    ) -> Dict[Union[NodeType, EdgeType], Any]:
        r"""Collects the attribute `key` from all node and edge types.

        Args:
            key (str): The attribute key to collect.
            exlude_None (bool, optional): If set to `True`, will exclude
                the `None` attribute values.
                (default: `False`)

        Example:
            >>> data = HeteroGraphData()
            >>> data['paper'].x = ...
            >>> data['author'].x = ...
            >>> data['author', 'writes', 'paper'].edge_index = ...
            >>> data.collect_attr('x')
            {'paper': ..., 'author': ...}
        """
        out = {}
        for _type, store in chain(self._nodes.items(), self._edges.items()):
            if hasattr(store, key):
                if exlude_None and getattr(store, key) is None:
                    continue
                out[_type] = getattr(store, key)
        return out

    def to_csc_dict(
        self,
        device: Optional[torch.device] = None,
        share_memory: bool = False,
        is_sorted: bool = False,
        node_time_d: Optional[Dict[NodeType, Tensor]] = None,
        edge_time_d: Optional[Dict[EdgeType, Tensor]] = None,
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor], Dict[str, Optional[Tensor]]]:
        r"""Convert the heterogeneous graph edge into a CSC format for sampling.
        Returns dictionaries holding `colptr` and `row` indices as well as edge
        permutations for each edge type, respectively.

        Args:
            device (torch.device, optional): The device to move the tensors to.
            share_memory (bool, optional): If set to `True`, will share memory
                with the original tensor.This can accelerate process when using
                multiple processes.
            is_sorted (bool, optional): If set to `True`, will not sort the edge
                index by column.
            node_time_d (Dict[str, Tensor], optional): The node time attribute
                dictionary.
            edge_time_d (Dict[str, Tensor], optional): The edge time attribute
                dictionary.

        Returns:
            - `colptr_d` holds the column pointers for each edge type.
            - `row_d` holds the row indices for each edge type.
            - `perm_d` holds the permutation indices for each edge type.
        """
        col_ptr_d, row_d, perm_d = {}, {}, {}

        for edge_type, store in self._edges.items():
            store: EdgeStorage
            src_node_time = (node_time_d or {}).get(edge_type[0], None)
            edge_time = (edge_time_d or {}).get(edge_type, None)
            out = store.to_csc(
                device=device,
                num_nodes=self[edge_type[0]].num_nodes,
                share_memory=share_memory,
                is_sorted=is_sorted,
                src_node_time=src_node_time,
                edge_time=edge_time,
            )
            col_ptr_d[edge_type] = out[0]
            row_d[edge_type] = out[1]
            perm_d[edge_type] = out[2]

        return col_ptr_d, row_d, perm_d

    def set_value_dict(
        self, key: str, value_d: Dict[Union[NodeType, EdgeType], Any]
    ) -> None:
        r"""Set the attribute `key` for each node and edge type in value dict.

        Args:
            key (str): The attribute key to set.
            value (Dict[Union[NodeType, EdgeType], Any]): The attribute values.
        """
        for type_, value in value_d.items():
            self[type_][key] = value

    # Dunder functions ########################################

    def __copy__(self):
        r"""Performs a shallow copy of the graph.

        1. Copy the properties and private attributes.
        2. Copy the `_mapping` storage (normal class attribute).
        3. Copy the `_nodes` and `_edges` dict.
            Copy the node and edge keys and storages.

        Storage copy is done by `copy.copy` which is a shallow copy,
        i.e. keeps the reference of the original tensor or other objects.
        """
        out = self.__class__.__new__(self.__class__)
        for k, v in self.__dict__.items():
            out.__dict__[k] = v
        out.__dict__["_mapping"] = copy.copy(self._mapping)
        out._mapping._parent = out
        out.__dict__["_nodes"] = {}
        out.__dict__["_edges"] = {}
        for k, v in self._nodes.items():
            out._nodes[k] = copy.copy(v)
            out._nodes[k]._parent = out
        for k, v in self._edges.items():
            out._edges[k] = copy.copy(v)
            out._edges[k]._parent = out
        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"num_nodes={self.num_nodes}, \n"
            f"node_types={self.node_types}, \n"
            f"edge_types={self.edge_types})\n"
        )
