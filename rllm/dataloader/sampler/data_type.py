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