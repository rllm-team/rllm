from typing import Dict, Optional, Tuple, List

import torch
from torch import Tensor


NodeType = str
EdgeType = Tuple[str, str, str]


def _sample_uniform_without_replacement(
    row_start: int,
    row_end: int,
    count: int,
    device: torch.device,
) -> List[int]:
    r"""Return absolute edge indices sampled uniformly without replacement
    from the range :obj:`[row_start, row_end)`.

    Args:
        row_start (int): The start of the edge index range (inclusive).
        row_end (int): The end of the edge index range (exclusive).
        count (int): The number of indices to sample. If negative or
            greater than the population size, all indices are returned.
        device (torch.device): The device for intermediate tensor operations.

    Returns:
        List[int]: Sampled absolute edge indices.
    """
    population = row_end - row_start
    if population == 0 or count == 0:
        return []
    if count < 0 or count >= population:
        return list(range(row_start, row_end))

    # Uniform sampling without replacement.
    perm = torch.randperm(population, device=device)[:count]
    return (perm + row_start).tolist()


def _prepare_seed_nodes(
    seed_dict: Dict[NodeType, Tensor],
    node_time_dict: Optional[Dict[NodeType, Tensor]],
    seed_time_dict: Optional[Dict[NodeType, Tensor]],
    device: torch.device,
) -> Tuple[
    Dict[NodeType, List[Tuple[int, int]]],
    Dict[NodeType, Dict[Tuple[int, int], int]],
    Dict[NodeType, List[int]],
    Dict[NodeType, Tuple[int, int]],
    List[int]
]:
    r"""Prepare initial seed nodes and their temporal information."""
    node_types = list(seed_dict.keys())
    sampled_nodes: Dict[NodeType, List[Tuple[int, int]]] = {nt: [] for nt in node_types}
    node_index: Dict[NodeType, Dict[Tuple[int, int], int]] = {nt: {} for nt in node_types}
    num_sampled_nodes_per_hop: Dict[NodeType, List[int]] = {nt: [0] for nt in node_types}
    slice_dict: Dict[NodeType, Tuple[int, int]] = {nt: (0, 0) for nt in node_types}
    seed_times: List[int] = []
    
    batch_idx = 0
    for nt, seeds in seed_dict.items():
        if seeds.numel() == 0:
            continue
        seeds = seeds.to(device=device, dtype=torch.long)
        
        node_time = None
        if node_time_dict is not None and nt in node_time_dict:
            node_time = node_time_dict[nt].to(device=device)
        seed_time = None
        if seed_time_dict is not None and nt in seed_time_dict:
            seed_time = seed_time_dict[nt].to(device=device)

        for i in range(seeds.numel()):
            nid = int(seeds[i].item())
            sampled_nodes[nt].append((batch_idx, nid))
            node_index[nt][(batch_idx, nid)] = len(sampled_nodes[nt]) - 1

            if seed_time is not None:
                t = int(seed_time[i].item())
            elif node_time is not None:
                t = int(node_time[nid].item())
            else:
                t = 0
            seed_times.append(t)
            batch_idx += 1

        slice_dict[nt] = (0, len(sampled_nodes[nt]))
        num_sampled_nodes_per_hop[nt][0] = len(sampled_nodes[nt])
        
    return sampled_nodes, node_index, num_sampled_nodes_per_hop, slice_dict, seed_times


def _sample_one_hop(
    ell: int,
    edge_types: List[EdgeType],
    node_types: List[NodeType],
    rowptr_dict: Dict[EdgeType, Tensor],
    col_dict: Dict[EdgeType, Tensor],
    num_neighbors_dict: Dict[EdgeType, List[int]],
    node_time_dict: Optional[Dict[NodeType, Tensor]],
    edge_time_dict: Optional[Dict[EdgeType, Tensor]],
    seed_times: List[int],
    sampled_nodes: Dict[NodeType, List[Tuple[int, int]]],
    node_index: Dict[NodeType, Dict[Tuple[int, int], int]],
    slice_dict: Dict[NodeType, Tuple[int, int]],
    row_out: Dict[EdgeType, List[int]],
    col_out: Dict[EdgeType, List[int]],
    edge_id_out: Dict[EdgeType, List[int]],
    num_edges_per_hop: Dict[EdgeType, List[int]],
    device: torch.device,
    csc: bool,
) -> None:
    r"""Perform one hop of neighbor sampling across all edge types."""
    for et in edge_types:
        if csc:
            src_type = et[2]
            dst_type = et[0]
        else:
            src_type = et[0]
            dst_type = et[2]

        rowptr = rowptr_dict[et].to(device=device, dtype=torch.long)
        col = col_dict[et].to(device=device, dtype=torch.long)
        neighbors_per_hop = num_neighbors_dict[et]
        
        if ell >= len(neighbors_per_hop):
            num_edges_per_hop[et].append(0)
            continue
        count = int(neighbors_per_hop[ell])

        edge_time = None
        if edge_time_dict is not None and et in edge_time_dict:
            edge_time = edge_time_dict[et].to(device=device)
        dst_node_time = None
        if node_time_dict is not None and dst_type in node_time_dict:
            dst_node_time = node_time_dict[dst_type].to(device=device)

        src_nodes = sampled_nodes[src_type]
        begin, end = slice_dict[src_type]
        hop_edge_count = 0

        for i in range(begin, end):
            batch, src_nid = src_nodes[i]
            seed_t = seed_times[batch] if seed_times else 0

            row_start = int(rowptr[src_nid].item())
            row_end = int(rowptr[src_nid + 1].item())

            if edge_time is not None:
                times = edge_time[row_start:row_end]
                if times.numel() == 0:
                    continue
                mask = times <= seed_t
                if not bool(mask.any()):
                    continue
                valid_idx = mask.nonzero(as_tuple=False).view(-1)
                last_valid = int(valid_idx[-1].item())
                row_end_eff = row_start + last_valid + 1
            elif dst_node_time is not None:
                neigh = col[row_start:row_end]
                if neigh.numel() == 0:
                    continue
                times = dst_node_time[neigh]
                mask = times <= seed_t
                if not bool(mask.any()):
                    continue
                valid_idx = mask.nonzero(as_tuple=False).view(-1)
                last_valid = int(valid_idx[-1].item())
                row_end_eff = row_start + last_valid + 1
            else:
                row_end_eff = row_end

            if row_end_eff <= row_start or count == 0:
                continue

            sampled_edge_indices = _sample_uniform_without_replacement(
                row_start=row_start,
                row_end=row_end_eff,
                count=count,
                device=device,
            )
            if not sampled_edge_indices:
                continue

            src_local_idx = node_index[src_type][(batch, src_nid)]

            for edge_id in sampled_edge_indices:
                dst_nid = int(col[edge_id].item())
                key = (batch, dst_nid)

                if key in node_index[dst_type]:
                    dst_local_idx = node_index[dst_type][key]
                else:
                    dst_local_idx = len(sampled_nodes[dst_type])
                    sampled_nodes[dst_type].append(key)
                    node_index[dst_type][key] = dst_local_idx

                row_out[et].append(src_local_idx)
                col_out[et].append(dst_local_idx)
                edge_id_out[et].append(edge_id)
                hop_edge_count += 1

        num_edges_per_hop[et].append(hop_edge_count)


def _build_output_tensors(
    edge_types: List[EdgeType],
    node_types: List[NodeType],
    sampled_nodes: Dict[NodeType, List[Tuple[int, int]]],
    row_out: Dict[EdgeType, List[int]],
    col_out: Dict[EdgeType, List[int]],
    edge_id_out: Dict[EdgeType, List[int]],
    device: torch.device,
    csc: bool,
) -> Tuple[
    Dict[EdgeType, Tensor],
    Dict[EdgeType, Tensor],
    Dict[NodeType, Tensor],
    Dict[NodeType, Tensor],
    Dict[EdgeType, Tensor]
]:
    r"""Convert lists of sampled indices into output PyTorch tensors."""
    node_id_dict: Dict[NodeType, Tensor] = {}
    batch_dict: Dict[NodeType, Tensor] = {}
    for nt in node_types:
        nodes = sampled_nodes.get(nt, [])
        if nodes:
            batches, ids = zip(*nodes)
            node_id_dict[nt] = torch.tensor(ids, device=device, dtype=torch.long)
            batch_dict[nt] = torch.tensor(batches, device=device, dtype=torch.long)
        else:
            node_id_dict[nt] = torch.empty(0, device=device, dtype=torch.long)
            batch_dict[nt] = torch.empty(0, device=device, dtype=torch.long)

    row_dict_tensor: Dict[EdgeType, Tensor] = {}
    col_dict_tensor: Dict[EdgeType, Tensor] = {}
    edge_id_dict_tensor: Dict[EdgeType, Tensor] = {}
    
    for et in edge_types:
        if row_out[et]:
            r = torch.tensor(row_out[et], device=device, dtype=torch.long)
            c = torch.tensor(col_out[et], device=device, dtype=torch.long)

            if csc:
                row_dict_tensor[et] = c
                col_dict_tensor[et] = r
            else:
                row_dict_tensor[et] = r
                col_dict_tensor[et] = c

            edge_id_dict_tensor[et] = torch.tensor(
                edge_id_out[et],
                device=device,
                dtype=torch.long,
            )
        else:
            row_dict_tensor[et] = torch.empty(0, device=device, dtype=torch.long)
            col_dict_tensor[et] = torch.empty(0, device=device, dtype=torch.long)
            edge_id_dict_tensor[et] = torch.empty(0, device=device, dtype=torch.long)
            
    return row_dict_tensor, col_dict_tensor, node_id_dict, batch_dict, edge_id_dict_tensor


def hetero_neighbor_sample_cpu(
    rowptr_dict: Dict[EdgeType, Tensor],
    col_dict: Dict[EdgeType, Tensor],
    seed_dict: Dict[NodeType, Tensor],
    num_neighbors_dict: Dict[EdgeType, List[int]],
    node_time_dict: Optional[Dict[NodeType, Tensor]] = None,
    edge_time_dict: Optional[Dict[EdgeType, Tensor]] = None,
    seed_time_dict: Optional[Dict[NodeType, Tensor]] = None,
    temporal_strategy: str = "uniform",
    csc: bool = False,
) -> Tuple[
    Dict[EdgeType, Tensor],
    Dict[EdgeType, Tensor],
    Dict[NodeType, Tensor],
    Dict[NodeType, Tensor],
    Optional[Dict[EdgeType, Tensor]],
    Dict[NodeType, List[int]],
    Dict[EdgeType, List[int]],
]:
    r"""Pure Python/CPU implementation of heterogeneous neighbor sampling with
    (optionally) temporal *uniform* sampling.

    - Multi-hop sampling over a heterogeneous graph (multiple edge types).
    - Per-relation, per-hop neighbor limits given by :obj:`num_neighbors_dict`.
    - Optional temporal constraint on either node times or edge times.
    - Within the temporally valid neighborhood, neighbors are sampled
      *uniformly without replacement*.

    Assumptions / limitations (for simplicity and clarity):

    - Graph is stored in **CSR** per edge type:
      :obj:`rowptr_dict[(src, rel, dst)]` is of size :obj:`[#src_nodes + 1]`
      and :obj:`col_dict[(src, rel, dst)]` has length equal to number of edges
      of that relation.
    - Only :obj:`temporal_strategy="uniform"` is implemented.
    - Sampling is always without replacement and directed.
    - For temporal sampling we create *disjoint* trees per seed; internally we
      track a batch index for each seed (like the C++ kernel), but we only
      return the global node indices.

    Args:
        rowptr_dict (Dict[EdgeType, Tensor]): CSR/CSC column pointers per
            edge type, of shape :obj:`[num_dst_nodes + 1]`.
        col_dict (Dict[EdgeType, Tensor]): Row indices (neighbor node ids)
            per edge type.
        seed_dict (Dict[NodeType, Tensor]): Seed node indices per node type
            to start sampling from.
        num_neighbors_dict (Dict[EdgeType, List[int]]): Number of neighbors
            to sample per hop for each edge type.
        node_time_dict (Dict[NodeType, Tensor], optional): Node-level
            timestamps for temporal filtering. (default: :obj:`None`)
        edge_time_dict (Dict[EdgeType, Tensor], optional): Edge-level
            timestamps for temporal filtering. (default: :obj:`None`)
        seed_time_dict (Dict[NodeType, Tensor], optional): Timestamps
            associated with each seed node. (default: :obj:`None`)
        temporal_strategy (str): Sampling strategy for temporal graphs.
            Currently only :obj:`'uniform'` is supported.
            (default: :obj:`'uniform'`)
        csc (bool): If :obj:`True`, assumes :obj:`rowptr_dict` and
            :obj:`col_dict` are in CSC format, i.e., traversal goes from
            destination to source. (default: :obj:`False`)

    Returns:
        Tuple: A 7-tuple of
        :obj:`(row_dict, col_dict, node_id_dict, batch_dict,
        edge_id_dict, num_sampled_nodes_per_hop, num_edges_per_hop)`.
    """
    if temporal_strategy != "uniform":
        raise ValueError("Only temporal_strategy='uniform' is supported.")
    if node_time_dict is not None and edge_time_dict is not None:
        raise ValueError(
            "Only one of node_time_dict or edge_time_dict may be specified.",
        )

    # -------------------------------------------------------------------------
    # Normalize and basic checks
    # -------------------------------------------------------------------------
    edge_types: List[EdgeType] = list(rowptr_dict.keys())
    node_types: List[NodeType] = sorted(
        set(seed_dict.keys())
        | {et[0] for et in edge_types}
        | {et[2] for et in edge_types}
    )

    if any(et not in col_dict for et in edge_types):
        missing = [et for et in edge_types if et not in col_dict]
        raise KeyError(f"Missing col_dict entries for edge_types: {missing}")
    if any(et not in num_neighbors_dict for et in edge_types):
        missing = [et for et in edge_types if et not in num_neighbors_dict]
        raise KeyError(
            f"Missing num_neighbors_dict entries for edge_types: {missing}",
        )

    # We assume all tensors live on the same (CPU) device:
    some_tensor = next(iter(rowptr_dict.values()))
    if some_tensor.is_cuda:
        raise ValueError("This implementation is CPU-only.")
    device = some_tensor.device

    # -------------------------------------------------------------------------
    # Prepare per-node-type storage: sampled nodes + index mapping
    # -------------------------------------------------------------------------
    # Ensure all nodes types are initialized in the dictionaries
    seed_dict_full = {nt: seed_dict.get(nt, torch.empty(0)) for nt in node_types}
    
    (
        sampled_nodes,
        node_index,
        num_sampled_nodes_per_hop,
        slice_dict,
        seed_times
    ) = _prepare_seed_nodes(
        seed_dict_full, node_time_dict, seed_time_dict, device
    )

    # -------------------------------------------------------------------------
    # Prepare per-edge-type storage for sampled edges and hop statistics
    # -------------------------------------------------------------------------
    row_out: Dict[EdgeType, List[int]] = {et: [] for et in edge_types}
    col_out: Dict[EdgeType, List[int]] = {et: [] for et in edge_types}
    edge_id_out: Dict[EdgeType, List[int]] = {et: [] for et in edge_types}
    num_edges_per_hop: Dict[EdgeType, List[int]] = {et: [] for et in edge_types}

    # Number of layers (hops): max length across all relations.
    L = max(len(v) for v in num_neighbors_dict.values()) if edge_types else 0

    # -------------------------------------------------------------------------
    # Multi-hop expansion
    # -------------------------------------------------------------------------
    for ell in range(L):
        _sample_one_hop(
            ell, edge_types, node_types, rowptr_dict, col_dict,
            num_neighbors_dict, node_time_dict, edge_time_dict,
            seed_times, sampled_nodes, node_index, slice_dict,
            row_out, col_out, edge_id_out, num_edges_per_hop,
            device, csc
        )

        # Update slices and per-hop node counts for next iteration.
        for nt in node_types:
            old_begin, old_end = slice_dict[nt]
            slice_dict[nt] = (old_end, len(sampled_nodes[nt]))
            num_sampled_nodes_per_hop[nt].append(
                slice_dict[nt][1] - slice_dict[nt][0],
            )

    # -------------------------------------------------------------------------
    # Build output tensors
    # -------------------------------------------------------------------------
    (
        row_dict_tensor,
        col_dict_tensor,
        node_id_dict,
        batch_dict,
        edge_id_dict_tensor
    ) = _build_output_tensors(
        edge_types, node_types, sampled_nodes, row_out, col_out, edge_id_out, device, csc
    )

    return (
        row_dict_tensor,
        col_dict_tensor,
        node_id_dict,
        batch_dict,
        edge_id_dict_tensor,
        num_sampled_nodes_per_hop,
        num_edges_per_hop,
    )
