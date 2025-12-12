from typing import List, Optional, Dict, Tuple

import torch
from torch import Tensor

from rllm.data import HeteroGraphData
from rllm.dataloader.sampler.utils import (
    convert_hdata_to_csc,
    hetero_neighbor_sample_cpu
)
from rllm.dataloader.sampler.data_type import (
    NodeSamplerInput,
    HeteroSamplerOutput
)


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
    ):

        assert device is None or device.type == 'cpu', 'Device must be CPU-enabled or None.'
        assert temporal_strategy == 'uniform', 'Only uniform temporal strategy is supported for now.'
        if temporal_strategy == 'uniform' and time_attr is None:
            raise ValueError('Time attribute must be provided for uniform temporal strategy.')

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

        self.num_neighbors = num_neighbors
        self.num_neighbors_dict = self._get_num_neighbor_dict()
        self.replace = replace
        self.temporal_strategy = temporal_strategy
        self.to_bidirectional = to_bidirectional

    def _get_num_neighbor_dict(self) -> Dict[Tuple[str, str, str], List[int]]:
        num_neighbors_dict = {}
        for etype in self.edge_types:
            num_neighbors_dict[etype] = self.num_neighbors
        return num_neighbors_dict

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

