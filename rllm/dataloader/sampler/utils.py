from typing import Optional, Dict

import torch
from torch import Tensor

from rllm.data import HeteroGraphData, EdgeStorage


def convert_hdata_to_csc(
    hdata: HeteroGraphData,
    device: Optional[torch.device] = None,
    share_memory: bool = False,
    is_sorted: bool = False,
    node_time_dict: Optional[Dict[str, Tensor]] = None,
    edge_time_dict: Optional[Dict[str, Tensor]] = None,
):
    r"""Convert a heterogeneous graph to CSC format for neighbor sampling.

    Args:
        hdata (HeteroGraphData): The heterogeneous graph data.
        device (torch.device, optional): The device to place output tensors on.
            (default: :obj:`None`)
        share_memory (bool): Whether to move tensors to shared memory for
            multi-process data loading. (default: :obj:`False`)
        is_sorted (bool): Whether the edge indices are already sorted by
            destination node. (default: :obj:`False`)
        node_time_dict (Dict[str, Tensor], optional): Node-level timestamps
            used for temporal filtering. (default: :obj:`None`)
        edge_time_dict (Dict[str, Tensor], optional): Edge-level timestamps
            used for temporal filtering. (default: :obj:`None`)

    Returns:
        Tuple[Dict, Dict, Dict]: A tuple of
        :obj:`(col_ptr_dict, row_dict, perm_dict)` where
        :obj:`col_ptr_dict` contains the column pointers,
        :obj:`row_dict` contains the row indices, and
        :obj:`perm_dict` contains the edge permutations, all keyed by
        edge type.
    """
    col_ptr_dict = {}
    row_dict = {}
    perm_dict = {}

    for edge_type, edge_store in hdata.edge_items():
        src_node_time = (node_time_dict or {}).get(edge_type[0], None)
        edge_time = (edge_time_dict or {}).get(edge_type, None)
        dst_node_type = edge_type[2]
        dst_num_nodes = hdata[dst_node_type].num_nodes

        edge_store: EdgeStorage
        out = edge_store.to_csc(
            device=device,
            num_nodes=dst_num_nodes,
            share_memory=share_memory,
            is_sorted=is_sorted,
            src_node_time=src_node_time,
            edge_time=edge_time,
        )
        col_ptr_dict[edge_type] = out[0]
        row_dict[edge_type] = out[1]
        perm_dict[edge_type] = out[2]

    return col_ptr_dict, row_dict, perm_dict
