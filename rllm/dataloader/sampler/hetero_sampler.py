from typing import List, Optional, Dict, Tuple
import warnings

import torch
from torch import Tensor

from rllm.data import HeteroGraphData
from rllm.dataloader.sampler.utils import (
    convert_hdata_to_csc,
    hetero_neighbor_sample_cpu
)
from rllm.dataloader.sampler.data_type import (
    NodeSamplerInput,
    HeteroSamplerOutput,
    NumNeighbors,
)
from rllm.utils import remap_dict_keys
import rllm.utils._pyglib


class HeteroSampler:
    """
    Heterogeneous graph sampler.

    Args:
        hdata (HeteroGraphData): The heterogeneous graph data.
        num_neighbors (List[int]): Number of neighbors to sample at each hop.
        replace (bool): Whether to sample with replacement. Default is False.
        temporal_strategy (str): Temporal sampling strategy.
            Currently only 'uniform' is supported.
        time_attr (Optional[str]): Node attribute name for time.
            Required if temporal_strategy is 'uniform'.
        device (Optional[torch.device]): Device to perform sampling on.
            Currently only CPU is supported.
        to_bidirectional (bool): Whether to convert the graph to bidirectional
            by adding reverse edges. Default is False.
        csc (bool): Whether to use CSC format for sampling. Default is False.
        use_pyg_lib (bool): Whether to use PyG-lib for sampling. Default is True.
    """
    def __init__(
        self,
        hdata: HeteroGraphData,
        num_neighbors: List[int],
        replace: bool = False,
        temporal_strategy: str = 'uniform',
        time_attr: Optional[str] = None,
        device: Optional[torch.device] = None,
        to_bidirectional: bool = False,
        csc: bool = False,
        use_pyg_lib: bool = True,
    ):

        assert device is None or device.type == 'cpu', 'Device must be CPU-enabled or None.'
        assert temporal_strategy == 'uniform', 'Only uniform temporal strategy is supported for now.'
        if temporal_strategy == 'uniform' and time_attr is None:
            raise ValueError('Time attribute must be provided for uniform temporal strategy.')

        if use_pyg_lib and rllm.utils._pyglib.WITH_PYG_LIB:
            self.use_pyglib = True
        else:
            if use_pyg_lib:
                warnings.warn("PyG-lib is not installed. Falling back to pure Python sampler.")
            self.use_pyglib = False

        self.csc = csc
        self.device = device or torch.device('cpu')
        self.node_types = hdata.node_types
        self.edge_types = hdata.edge_types
        self.num_nodes = {
            node_type: hdata[node_type].num_nodes
            for node_type in hdata.node_types
        }

        self.node_time_dict = hdata.collect_attr(time_attr)

        (
            self.col_ptr_dict,
            self.row_dict,
            self.perm_dict
        ) = convert_hdata_to_csc(
            hdata=hdata,
            device=self.device,
            share_memory=True,
            is_sorted=False,
            node_time_dict=self.node_time_dict,
            edge_time_dict=None,
        )

        # Only convert edge type keys to strings when we actually use pyg-lib.
        # The pure Python sampler (`hetero_neighbor_sample_cpu`) expects
        # edge type keys to be tuples of (src, rel, dst).
        if self.use_pyglib:
            # Pyg_lib sampler requires the edge types to be a string.
            # Convert the edge types from tuple to string.
            self.to_rel_type = {k: '__'.join(k) for k in self.edge_types}
            self.to_edge_type = {v: k for k, v in self.to_rel_type.items()}

            self.row_dict = remap_dict_keys(self.row_dict, self.to_rel_type)
            self.col_ptr_dict = remap_dict_keys(self.col_ptr_dict, self.to_rel_type)

        self.num_neighbors = num_neighbors
        self.replace = replace
        self.temporal_strategy = temporal_strategy
        self.disjoint = True
        self.to_bidirectional = to_bidirectional

        if not self.use_pyglib:
            self.num_neighbors_dict = self._get_num_neighbor_dict()

    # num_neighbors
    @property
    def num_neighbors(self) -> NumNeighbors:
        return self._num_neighbors

    @num_neighbors.setter
    def num_neighbors(self, num_neighbors: List[int]):
        self._num_neighbors = NumNeighbors(num_neighbors)

    def _get_num_neighbor_dict(self) -> Dict[Tuple[str, str, str], List[int]]:
        num_neighbors_dict = self.num_neighbors.get_values(self.edge_types)
        return num_neighbors_dict

    # is_temporal
    @property
    def is_temporal(self) -> bool:
        return self.node_time_dict is not None

    # disjoint
    # disjoint is set to True if temporal sampling is enabled.
    @property
    def disjoint(self) -> bool:
        return self._disjoint or self.is_temporal

    @disjoint.setter
    def disjoint(self, disjoint: bool):
        self._disjoint = disjoint

    @property
    def edge_permutation(self) -> Dict[Tuple[str, str, str], Tensor]:
        return self.perm_dict

    def sample_neighbors(self, input: NodeSamplerInput) -> HeteroSamplerOutput:
        seed = {input.input_type: input.node}
        seed_time = None
        if input.time is not None:
            seed_time = {input.input_type: input.time}

        out: HeteroSamplerOutput = self._sample_neighbors(seed, seed_time)
        out.metadata = (input.input_id, input.time)

        if self.to_bidirectional:
            out = out.to_bidirectional()
        return out

    def _sample_neighbors(
        self,
        seed: Dict[str, Tensor],
        seed_time: Optional[Dict[str, Tensor]]
    ) -> HeteroSamplerOutput:

        if self.use_pyglib:
            colptrs = list(self.col_ptr_dict.values())
            dtype = colptrs[0].dtype if len(colptrs) > 0 else torch.int64
            seed = {k: v.to(dtype) for k, v in seed.items()}

            args = (
                self.node_types,
                self.edge_types,
                self.col_ptr_dict,
                self.row_dict,
                seed,
                self.num_neighbors.get_mapped_values(self.edge_types),
                self.node_time_dict,
            )
            args += (None, )    # edge time
            args += (seed_time, )
            args += (None, )    # edge weight
            args += (
                True,   # csc format
                self.replace,
                True,  # subgraph type
                self.disjoint,
                self.temporal_strategy,
                True,   # return edge id
            )

            out = torch.ops.pyg.hetero_neighbor_sample(*args)

            row, col, node, edge, batch = out[:4] + (None, )
            # `pyg-lib>0.1.0` returns sampled number of nodes/edges:
            num_sampled_nodes = num_sampled_edges = None
            if len(out) >= 6:
                num_sampled_nodes, num_sampled_edges = out[4:6]

            if self.disjoint:
                node = {k: v.t().contiguous() for k, v in node.items()}
                batch = {k: v[0] for k, v in node.items()}
                node = {k: v[1] for k, v in node.items()}

            # remap the edge type
            row = remap_dict_keys(row, self.to_edge_type)
            col = remap_dict_keys(col, self.to_edge_type)
            # edge = remap_keys(edge, self.to_edge_type)

            if num_sampled_edges is not None:
                num_sampled_edges = remap_dict_keys(
                    num_sampled_edges,
                    self.to_edge_type,
                )

            return HeteroSamplerOutput(
                node=node,
                row=row,
                col=col,
                batch=batch,
                num_sampled_nodes=num_sampled_nodes,
                num_sampled_edges=num_sampled_edges,
            )

        else:
            """
            We do sample for each target node, i.e., for each column.
            So we use col_ptr_dict as rowptr.

            The sampling is:
            - directed,
            - temporal uniform,
            - disjoint per seed node and
            - not replaced.
            """
            (
                row_dict,
                col_dict,
                node_id_dict,
                batch_dict,
                _,
                num_sampled_nodes_per_hop,
                num_edges_per_hop
            ) = hetero_neighbor_sample_cpu(
                rowptr_dict=self.col_ptr_dict,
                col_dict=self.row_dict,
                seed_dict=seed,
                num_neighbors_dict=self.num_neighbors_dict,
                node_time_dict=self.node_time_dict,
                edge_time_dict=None,
                seed_time_dict=seed_time,
                temporal_strategy=self.temporal_strategy,
                csc=True,
            )

            return HeteroSamplerOutput(
                node=node_id_dict,
                row=row_dict,
                col=col_dict,
                batch=batch_dict,
                num_sampled_nodes=num_sampled_nodes_per_hop,
                num_sampled_edges=num_edges_per_hop,
                original_row=None,
                original_col=None,
                metadata=None,
            )

