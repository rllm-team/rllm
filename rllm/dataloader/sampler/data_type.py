from typing import Any, Optional, Union, Dict, Tuple, List
import copy
from collections import defaultdict
from dataclasses import dataclass
import warnings

import torch
from torch import Tensor

from rllm.utils._mixin import CastMixin
from rllm.utils.graph_utils import to_bidirectional


@dataclass(init=False)
class NodeSamplerInput(CastMixin):
    r"""The sampling input data class.

    Args:
        input_id (torch.Tensor, optional): The indices of the data loader input
            of the current mini-batch.
        node (torch.Tensor): The indices of seed nodes to start sampling from.
        time (torch.Tensor, optional): The timestamp for the seed nodes.
            (default: :obj:`None`)
        input_type (str, optional): The input node type (in case of sampling in
            a heterogeneous graph). (default: :obj:`None`)
    """
    input_id: Optional[Tensor]
    node: Tensor
    time: Optional[Tensor] = None
    input_type: Optional[str] = None

    def __init__(
        self,
        input_id: Optional[Tensor],
        node: Tensor,
        time: Optional[Tensor] = None,
        input_type: Optional[str] = None,
    ):
        if input_id is not None:
            input_id = input_id.cpu()
        node = node.cpu()
        if time is not None:
            time = time.cpu()

        self.input_id = input_id
        self.node = node
        self.time = time
        self.input_type = input_type

    def __getitem__(self, index: Union[Tensor, Any]) -> 'NodeSamplerInput':
        if not isinstance(index, Tensor):
            index = torch.tensor(index, dtype=torch.long)

        return NodeSamplerInput(
            self.input_id[index] if self.input_id is not None else index,
            self.node[index],
            self.time[index] if self.time is not None else None,
            self.input_type,
        )


@dataclass
class HeteroSamplerOutput(CastMixin):
    """
    Outout of the heterosampler, which only contains
    the indices of the sampled nodes and edges.

    node (Dict[str, Tensor]): The indices of the sampled nodes.
    row (Dict[Tuple, Tensor]): The row indices of the sampled edges.
    col (Dict[Tuple, Tensor]): The column indices of the sampled edges.
    batch (Dict[str, Tensor]): Not used yet.
    num_sampled_nodes (Dict[str, List[int]]): The number of sampled nodes
        for each node type in each hop.
    num_sampled_edges (Dict[Tuple, List[int]]): The number of sampled edges
        for each edge type in each hop.

    Edge type: Tuple[str, str, str]
    Node type: str
    """
    node: Dict[str, Tensor]
    row: Dict[Tuple, Tensor]
    col: Dict[Tuple, Tensor]
    batch: Optional[Dict[str, Tensor]] = None
    num_sampled_nodes: Optional[Dict[str, List[int]]] = None
    num_sampled_edges: Optional[Dict[Tuple, List[int]]] = None
    original_row: Optional[Dict[Tuple, Tensor]] = None
    original_col: Optional[Dict[Tuple, Tensor]] = None
    metadata: Optional[Any] = None

    def to_bidirectional(self, keep_org=False) -> 'HeteroSamplerOutput':
        r"""Converts the sampled subgraph into a bidirectional variant, in
        which all sampled edges are guaranteed to be bidirectional.

        Args:
            keep_org (bool): Whether to keep the original edges in
                `original_row` and `original_col`. Default is False.
        """
        out = copy.copy(self)
        out.row = copy.copy(self.row)
        out.col = copy.copy(self.col)

        if keep_org:
            out.original_row = {}
            out.original_col = {}
            for key in self.row.keys():
                out.original_row[key] = self.row[key]
                out.original_col[key] = self.col[key]
        else:
            out.original_row = out.original_col = None

        src_dst_dict = defaultdict(list)
        edge_types = self.row.keys()
        edge_types = [k for k in edge_types if not k[1].startswith('rev_')]

        for edge_type in edge_types:
            src, rel, dst = edge_type
            rev_edge_type = (dst, f'rev_{rel}', src)

            if src == dst and rev_edge_type not in self.row:
                out.row[edge_type], out.col[edge_type] = to_bidirectional(
                    row=self.row[edge_type],
                    col=self.col[edge_type],
                    rev_row=self.row[edge_type],
                    rev_col=self.col[edge_type],
                )

            elif rev_edge_type in self.row:
                out.row[edge_type], out.col[edge_type] = to_bidirectional(
                    row=self.row[edge_type],
                    col=self.col[edge_type],
                    rev_row=self.row[rev_edge_type],
                    rev_col=self.col[rev_edge_type],
                )
                out.row[rev_edge_type] = out.col[edge_type]
                out.col[rev_edge_type] = out.row[edge_type]

            else:  # Find the reverse edge type (if it is unique):
                if len(src_dst_dict) == 0:  # Create mapping lazily.
                    for key in self.row.keys():
                        v1, _, v2 = key
                        src_dst_dict[(v1, v2)].append(key)

                if len(src_dst_dict[(dst, src)]) == 1:
                    rev_edge_type = src_dst_dict[(dst, src)][0]
                    row, col, _ = to_bidirectional(
                        row=self.row[edge_type],
                        col=self.col[edge_type],
                        rev_row=self.row[rev_edge_type],
                        rev_col=self.col[rev_edge_type],
                    )
                    out.row[edge_type] = row
                    out.col[edge_type] = col

                else:
                    warnings.warn(f"Cannot convert to bidirectional graph "
                                  f"since the edge type {edge_type} does not "
                                  f"seem to have a reverse edge type")

        return out


EdgeType = Tuple[str, str, str]


class EdgeTypeStr(str):
    r"""A helper class to construct serializable edge types by merging an edge
    type tuple into a single string.
    """
    EDGE_TYPE_STR_SPLIT = '__'
    DEFAULT_REL = 'to'
    edge_type: tuple[str, str, str]

    def __new__(cls, *args: Any) -> 'EdgeTypeStr':
        if isinstance(args[0], (list, tuple)):
            # Unwrap `EdgeType((src, rel, dst))` and `EdgeTypeStr((src, dst))`:
            args = tuple(args[0])

        if len(args) == 1 and isinstance(args[0], str):
            arg = args[0]  # An edge type string was passed.
            edge_type = tuple(arg.split(cls.EDGE_TYPE_STR_SPLIT))
            if len(edge_type) != 3:
                raise ValueError(f"Cannot convert the edge type '{arg}' to a "
                                 f"tuple since it holds invalid characters")

        elif len(args) == 2 and all(isinstance(arg, str) for arg in args):
            # A `(src, dst)` edge type was passed - add `DEFAULT_REL`:
            edge_type = (args[0], cls.DEFAULT_REL, args[1])
            arg = cls.EDGE_TYPE_STR_SPLIT.join(edge_type)

        elif len(args) == 3 and all(isinstance(arg, str) for arg in args):
            # A `(src, rel, dst)` edge type was passed:
            edge_type = tuple(args)
            arg = cls.EDGE_TYPE_STR_SPLIT.join(args)

        else:
            raise ValueError(f"Encountered invalid edge type '{args}'")

        out = str.__new__(cls, arg)
        out.edge_type = edge_type  # type: ignore
        return out

    def to_tuple(self) -> EdgeType:
        r"""Returns the original edge type."""
        if len(self.edge_type) != 3:
            raise ValueError(f"Cannot convert the edge type '{self}' to a "
                             f"tuple since it holds invalid characters")
        return self.edge_type

    def __reduce__(self) -> tuple[Any, Any]:
        return (self.__class__, (self.edge_type, ))


@dataclass(frozen=True)
class NumNeighbors:
    r"""The number of neighbors to sample in a heterogeneous graph. 
    It may also take in a dictionary denoting
    the amount of neighbors to sample for individual edge types.

    Args:
        values (List[int] or Dict[Tuple[str, str, str], List[int]]): The
            number of neighbors to sample.
            If an entry is set to :obj:`-1`, all neighbors will be included.
            In heterogeneous graphs, may also take in a dictionary denoting
            the amount of neighbors to sample for individual edge types.
        default (List[int], optional): The default number of neighbors for edge
            types not specified in :obj:`values`. (default: :obj:`None`)
    """
    values: Union[List[int], Dict[EdgeTypeStr, List[int]]]
    default: Optional[List[int]] = None

    def __init__(
        self,
        values: Union[List[int], Dict[EdgeType, List[int]]],
        default: Optional[List[int]] = None,
    ):
        if isinstance(values, (tuple, list)) and default is not None:
            raise ValueError(f"'default' must be set to 'None' in case a "
                             f"single list is given as the number of "
                             f"neighbors (got '{type(default)})'")

        if isinstance(values, dict):
            values = {EdgeTypeStr(key): value for key, value in values.items()}

        # Write to `__dict__` since dataclass is annotated with `frozen=True`:
        self.__dict__['values'] = values
        self.__dict__['default'] = default

    def _get_values(
        self,
        edge_types: Optional[List[EdgeType]] = None,
        mapped: bool = False,
    ) -> Union[List[int], Dict[Union[EdgeType, EdgeTypeStr], List[int]]]:

        if edge_types is not None:
            if isinstance(self.values, (tuple, list)):
                default = self.values
            elif isinstance(self.values, dict):
                default = self.default
            else:
                raise AssertionError()

            # Confirm that `values` only hold valid edge types:
            if isinstance(self.values, dict):
                edge_types_str = {EdgeTypeStr(key) for key in edge_types}
                invalid_edge_types = set(self.values.keys()) - edge_types_str
                if len(invalid_edge_types) > 0:
                    raise ValueError("Not all edge types specified in "
                                     "'num_neighbors' exist in the graph")

            out = {}
            for edge_type in edge_types:
                edge_type_str = EdgeTypeStr(edge_type)
                if edge_type_str in self.values:
                    out[edge_type_str if mapped else edge_type] = (
                        self.values[edge_type_str])
                else:
                    if default is None:
                        raise ValueError(f"Missing number of neighbors for "
                                         f"edge type '{edge_type}'")
                    out[edge_type_str if mapped else edge_type] = default

        elif isinstance(self.values, dict) and not mapped:
            out = {key.to_tuple(): value for key, value in self.values.items()}

        else:
            out = copy.copy(self.values)

        if isinstance(out, dict):
            num_hops = {len(v) for v in out.values()}
            if len(num_hops) > 1:
                raise ValueError(f"Number of hops must be the same across all "
                                 f"edge types (got {len(num_hops)} different "
                                 f"number of hops)")

        return out

    def get_values(
        self,
        edge_types: Optional[List[EdgeType]] = None,
    ) -> Union[List[int], Dict[EdgeType, List[int]]]:
        r"""Returns the number of neighbors.

        Args:
            edge_types (List[Tuple[str, str, str]], optional): The edge types
                to generate the number of neighbors for. (default: :obj:`None`)
        """
        if '_values' in self.__dict__:
            return self.__dict__['_values']

        values = self._get_values(edge_types, mapped=False)

        self.__dict__['_values'] = values
        return values

    def get_mapped_values(
        self,
        edge_types: Optional[List[EdgeType]] = None,
    ) -> Union[List[int], Dict[str, List[int]]]:
        r"""Returns the number of neighbors.
        For heterogeneous graphs, a dictionary is returned in which edge type
        tuples are converted to strings.

        Args:
            edge_types (List[Tuple[str, str, str]], optional): The edge types
                to generate the number of neighbors for. (default: :obj:`None`)
        """
        if '_mapped_values' in self.__dict__:
            return self.__dict__['_mapped_values']

        values = self._get_values(edge_types, mapped=True)

        self.__dict__['_mapped_values'] = values
        return values

    @property
    def num_hops(self) -> int:
        r"""Returns the number of hops."""
        if '_num_hops' in self.__dict__:
            return self.__dict__['_num_hops']

        if isinstance(self.values, (tuple, list)):
            num_hops = max(len(self.values), len(self.default or []))
        else:  # isinstance(self.values, dict):
            num_hops = max([0] + [len(v) for v in self.values.values()])
            num_hops = max(num_hops, len(self.default or []))

        self.__dict__['_num_hops'] = num_hops
        return num_hops

    def __len__(self) -> int:
        r"""Returns the number of hops."""
        return self.num_hops
