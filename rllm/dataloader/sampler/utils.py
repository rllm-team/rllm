from typing import Dict, Optional, Tuple, TypeVar, Union, List, Any

import torch
from torch import Tensor

# from rllm.types import
from rllm.data import HeteroGraphData, EdgeStorage


def convert_hdata_to_csc(
    hdata: HeteroGraphData,
    device: Optional[torch.device] = None,
    share_memory: bool = False,
    is_sorted: bool = False,
    node_time_dict: Optional[Dict[str, Tensor]] = None,
    edge_time_dict: Optional[Dict[str, Tensor]] = None,
):
    """
    Convert the heterogeneous graph to a CSC format.

    Returns:
        col_ptr_dict: Dict[str, Tensor]
        row_dict: Dict[str, Tensor]
        perm_dict: Dict[str, Tensor]
    """
    col_ptr_dict = {}
    row_dict = {}
    perm_dict = {}

    for edge_type, edge_store in hdata.edge_items():
        src_node_time = (node_time_dict or {}).get(edge_type[0], None)
        edge_time     = (edge_time_dict or {}).get(edge_type, None)
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


X = TypeVar('X')
Y = TypeVar('Y')

def remap_keys(
    inputs: Dict[X, Any],
    mapping: Dict[X, Y],
    exclude: Optional[List[X]] = None,
) -> Dict[Union[X, Y], Any]:
    exclude = exclude or []
    return {
        k if k in exclude else mapping.get(k, k): v
        for k, v in inputs.items()
    }


# Hetero sampler backend
NodeType = str
EdgeType = Tuple[str, str, str]

def _sample_uniform_without_replacement(
    row_start: int,
    row_end: int,
    count: int,
    device: torch.device,
) -> List[int]:
    """Return absolute edge indices in [row_start, row_end)."""
    population = row_end - row_start
    if population == 0 or count == 0:
        return []
    if count < 0 or count >= population:
        return list(range(row_start, row_end))

    # Uniform sampling without replacement.
    perm = torch.randperm(population, device=device)[:count]
    return (perm + row_start).tolist()


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

    Return:
        row_dict: Dict[EdgeType, Tensor]
        col_dict: Dict[EdgeType, Tensor]
        node_id_dict: Dict[NodeType, Tensor]
        edge_id_dict: Optional[Dict[EdgeType, Tensor]]
        num_sampled_nodes_per_hop: Dict[NodeType, List[int]]
        num_edges_per_hop: Dict[EdgeType, List[int]]

    Args:
    csc (bool): If True, assumes `rowptr_dict` and `col_dict` are in CSC format.
        This means for an edge type (src, rel, dst), the traversal is performed
        from dst to src (reverse direction). The `rowptr_dict` indexes dst nodes,
        and `col_dict` contains src nodes.
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
    # Internally we store (batch_idx, global_node_id) for each sampled node.
    # -------------------------------------------------------------------------
    sampled_nodes: Dict[NodeType, List[Tuple[int, int]]] = {
        nt: [] for nt in node_types
    }
    node_index: Dict[NodeType, Dict[Tuple[int, int], int]] = {
        nt: {} for nt in node_types
    }
    num_sampled_nodes_per_hop: Dict[NodeType, List[int]] = {
        nt: [0] for nt in node_types
    }
    # slice_dict[nt] = (begin, end) indices into sampled_nodes[nt] for current hop.
    slice_dict: Dict[NodeType, Tuple[int, int]] = {
        nt: (0, 0) for nt in node_types
    }

    # Flatten seed timestamps per "batch" (root of each sampled tree).
    seed_times: List[int] = []
    batch_idx = 0

    for nt, seeds in seed_dict.items():
        if seeds.numel() == 0:
            continue
        seeds = seeds.to(device=device, dtype=torch.long)
        # Determine seed times for this node type, if any.
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
                # No temporal information; value is unused in static sampling.
                t = 0
            seed_times.append(t)
            batch_idx += 1

        slice_dict[nt] = (0, len(sampled_nodes[nt]))
        num_sampled_nodes_per_hop[nt][0] = len(sampled_nodes[nt])

    # -------------------------------------------------------------------------
    # Prepare per-edge-type storage for sampled edges and hop statistics
    # -------------------------------------------------------------------------
    row_out: Dict[EdgeType, List[int]] = {et: [] for et in edge_types}
    col_out: Dict[EdgeType, List[int]] = {et: [] for et in edge_types}
    edge_id_out: Dict[EdgeType, List[int]] = {et: [] for et in edge_types}
    num_edges_per_hop: Dict[EdgeType, List[int]] = {
        et: [] for et in edge_types
    }

    # Number of layers (hops): max length across all relations.
    L = max(len(v) for v in num_neighbors_dict.values()) if edge_types else 0

    # -------------------------------------------------------------------------
    # Multi-hop expansion
    # -------------------------------------------------------------------------
    for ell in range(L):
        # We first sample along each relation, accumulating newly found dst
        # nodes in tmp dict, and only then update global slices.
        dst_new_nodes: Dict[NodeType, List[Tuple[int, int]]] = {
            nt: [] for nt in node_types
        }

        for et in edge_types:
            if csc:
                # In CSC mode, we traverse from dst -> src (reverse edge).
                # rowptr indexes dst nodes. col contains src nodes.
                src_type = et[2]
                dst_type = et[0]
            else:
                src_type = et[0]
                dst_type = et[2]

            rowptr = rowptr_dict[et].to(device=device, dtype=torch.long)
            col = col_dict[et].to(device=device, dtype=torch.long)
            neighbors_per_hop = num_neighbors_dict[et]
            if ell >= len(neighbors_per_hop):
                # No expansion along this relation at this hop.
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

                # CSR neighborhood of this source node:
                row_start = int(rowptr[src_nid].item())
                row_end = int(rowptr[src_nid + 1].item())

                # Temporal filtering: restrict to edges satisfying time <= seed_t.
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

                # Uniform sampling within the temporally valid neighborhood.
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

                    # Map or insert dst node for this type:
                    if key in node_index[dst_type]:
                        dst_local_idx = node_index[dst_type][key]
                    else:
                        dst_local_idx = len(sampled_nodes[dst_type])
                        sampled_nodes[dst_type].append(key)
                        node_index[dst_type][key] = dst_local_idx
                        dst_new_nodes[dst_type].append(key)

                    row_out[et].append(src_local_idx)
                    col_out[et].append(dst_local_idx)
                    edge_id_out[et].append(edge_id)
                    hop_edge_count += 1

            num_edges_per_hop[et].append(hop_edge_count)

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
    node_id_dict: Dict[NodeType, Tensor] = {}
    batch_dict: Dict[NodeType, Tensor] = {}
    for nt, nodes in sampled_nodes.items():
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
                # In CSC mode, row_out contains indices of traversal source (original dst),
                # and col_out contains indices of traversal dest (original src).
                # We want to return row/col matching the original graph direction (src->dst).
                # So row_dict (src) gets col_out, col_dict (dst) gets row_out.
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
            edge_id_dict_tensor[et] = torch.empty(0, device=device,
                                                  dtype=torch.long)

    return (
        row_dict_tensor,
        col_dict_tensor,
        node_id_dict,
        batch_dict,
        edge_id_dict_tensor,
        num_sampled_nodes_per_hop,
        num_edges_per_hop,
    )