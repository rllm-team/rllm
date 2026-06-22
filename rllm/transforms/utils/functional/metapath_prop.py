from typing import Dict, Iterable, Optional, Tuple
from collections import defaultdict

import torch
from torch import Tensor

from rllm.utils.seg_reduce import seg_sum


EdgeType = Tuple[str, str, str]


def _rel_token(edge_type: EdgeType) -> str:
    src, rel, dst = edge_type
    return f"{src}-{rel}->{dst}"


def row_norm_edge_weight(
    edge_index: Tensor,
    num_src_nodes: int,
) -> Tensor:
    r"""Compute row-normalized edge weights :math:`1 / \deg(\text{src})`.

    Each edge is weighted by the inverse out-degree of its source node, so
    that messages aggregated at a destination node form a mean over its
    incoming neighbors.

    Args:
        edge_index (Tensor): Edge indices of shape :obj:`[2, num_edges]`.
        num_src_nodes (int): Number of source nodes.

    Returns:
        Tensor: Per-edge weights of shape :obj:`[num_edges]`.

    Example:
        >>> import torch
        >>> edge_index = torch.tensor([[0, 0, 1], [1, 2, 2]])
        >>> row_norm_edge_weight(edge_index, num_src_nodes=2)
        tensor([0.5000, 0.5000, 1.0000])
    """
    row = edge_index[0]
    deg = torch.bincount(row, minlength=num_src_nodes).to(torch.float)
    deg[deg == 0] = 1.0
    return (1.0 / deg[row]).to(edge_index.device)


def weighted_mean_aggregate(
    x_src: Tensor,
    edge_index: Tensor,
    edge_weight: Tensor,
    num_dst: int,
) -> Tensor:
    r"""Aggregate source features to destination nodes via a weighted mean.

    Computes, for each destination node :math:`i`:

    .. math::
        \mathbf{o}_i = \frac{\sum_{j} w_{j,i} \mathbf{x}_j}
        {\sum_{j} w_{j,i}}

    Args:
        x_src (Tensor): Source node features of shape :obj:`[num_src, F]`.
        edge_index (Tensor): Edge indices of shape :obj:`[2, num_edges]`.
        edge_weight (Tensor): Per-edge weights of shape :obj:`[num_edges]`.
        num_dst (int): Number of destination nodes.

    Returns:
        Tensor: Aggregated destination features of shape :obj:`[num_dst, F]`.
    """
    src, dst = edge_index[0], edge_index[1]
    msgs = x_src.index_select(0, src) * edge_weight.unsqueeze(-1)
    num = seg_sum(msgs, dst, num_dst)
    den = seg_sum(edge_weight.unsqueeze(-1), dst, num_dst).clamp(min=1e-12)
    return num / den


def metapath_propagate(
    edge_index_dict: Dict[Tuple[str, str, str], Tensor],
    x_dict: Dict[str, Tensor],
    num_nodes_dict: Dict[str, int],
    target_node_type: str,
    min_hops: int = 0,
    max_hops: int = 3,
    edge_schema: Optional[Iterable[EdgeType]] = None,
) -> Dict[str, Tensor]:
    r"""Generate multi-hop meta-path features for a target node type.

    Starting from base node features, this iteratively propagates features
    along each edge type using a row-normalized weighted mean. At each hop the
    new features are keyed by their meta-path (which carries the full
    ``(src, rel, dst)`` token of every traversed edge so parallel relations
    between the same pair of node types are not collapsed), and only
    target-type features (plus the latest-hop intermediates) are retained to
    bound memory.

    Args:
        edge_index_dict (Dict[Tuple[str, str, str], Tensor]): Edge indices by
            edge type. Edge types missing from this mapping (or empty) are
            still traversed against ``edge_schema`` with a zero feature so the
            output schema is stable across mini-batches.
        x_dict (Dict[str, Tensor]): Base node features by node type.
        num_nodes_dict (Dict[str, int]): Number of nodes by node type.
        target_node_type (str): The node type to collect meta-path features for.
        min_hops (int): Minimum hop count to keep. (default: :obj:`0`)
        max_hops (int): Maximum number of propagation hops. (default: :obj:`3`)
        edge_schema (Optional[Iterable[Tuple[str, str, str]]]): Full set of
            edge types to traverse. When provided, batches with missing edges
            still yield the same meta-paths (zero-filled), so the downstream
            fusion model sees a constant number of meta-paths :obj:`M`.

    Returns:
        Dict[str, Tensor]: Mapping from meta-path name to a feature tensor of
        shape :obj:`[num_target_nodes, 1, F]`. The values are ordered by key
        when concatenated by the caller.
    """
    if edge_schema is None:
        schema = list(edge_index_dict.keys())
    else:
        schema = list(edge_schema)

    feat_table: Dict[str, Dict[str, Tensor]] = {}
    for nt, feat in x_dict.items():
        if isinstance(feat, Tensor):
            feat_table[nt] = {f"0_{nt}": feat}
        else:
            feat_table[nt] = {}

    edge_weight_dict: Dict[Tuple[str, str, str], Tensor] = {}
    for edge_type in schema:
        edge_index = edge_index_dict.get(edge_type)
        if edge_index is None or edge_index.numel() == 0:
            edge_weight_dict[edge_type] = None  # type: ignore[assignment]
            continue
        src_type = edge_type[0]
        edge_weight_dict[edge_type] = row_norm_edge_weight(
            edge_index, num_nodes_dict[src_type]
        )

    for hop in range(1, max_hops + 1):
        for edge_type in schema:
            src_type, _, dst_type = edge_type
            rel_tok = _rel_token(edge_type)
            n_dst = num_nodes_dict[dst_type]

            for key, src_x in list(feat_table[src_type].items()):
                if not key.startswith(f"{hop - 1}_"):
                    continue
                new_key = key.replace(f"{hop - 1}_", f"{hop}_{rel_tok}.")
                edge_index = edge_index_dict.get(edge_type)
                edge_weight = edge_weight_dict.get(edge_type)
                if (
                    edge_index is None
                    or edge_index.numel() == 0
                    or edge_weight is None
                ):
                    # Schema-stable zero fill so M is constant across batches.
                    new_dst_x = src_x.new_zeros((n_dst, src_x.size(-1)))
                else:
                    new_dst_x = weighted_mean_aggregate(
                        x_src=src_x,
                        edge_index=edge_index,
                        edge_weight=edge_weight,
                        num_dst=n_dst,
                    )
                feat_table[dst_type][new_key] = new_dst_x

        # Only keep newly generated features and target-type features.
        for nt, feat_d in feat_table.items():
            if nt == target_node_type:
                continue
            rm_feats = [k for k in feat_d if not k.startswith(f"{hop}_")]
            for k in rm_feats:
                del feat_d[k]

    grouped: Dict[str, list] = defaultdict(list)
    for key, feat in feat_table[target_node_type].items():
        if int(key.split("_", 1)[0]) < min_hops:
            continue
        grouped[key].append(feat)
    if len(grouped) == 0:
        raise RuntimeError(
            f"No features with hop >= {min_hops} were generated "
            f"for '{target_node_type}'."
        )
    meta_path_feats: Dict[str, Tensor] = {
        key: torch.stack(feats, dim=1) for key, feats in grouped.items()
    }
    return meta_path_feats
