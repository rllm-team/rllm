from typing import Optional

import torch
from torch import Tensor

from rllm.data import TableData, NodeStorage, EdgeStorage


def filter_edge_store_(
    store: EdgeStorage,
    out_store: EdgeStorage,
    row: Tensor,
    col: Tensor,
    perm: Optional[Tensor] = None
):
    r"""Filter an edge storage to only hold the sampled edges
    represented by :obj:`(row, col)`.

    Args:
        store (EdgeStorage): The source edge storage.
        out_store (EdgeStorage): The output edge storage to write into.
        row (Tensor): Source node indices of sampled edges.
        col (Tensor): Destination node indices of sampled edges.
        perm (Tensor, optional): Edge permutation indices.
            (default: :obj:`None`)
    """
    for key, value in store.items():
        if key == 'edge_index':
            edge_index = torch.stack([row, col], dim=0).to(value.device)
            out_store.edge_index = edge_index
        else:
            raise NotImplementedError(
                f"Edge attribute key: {key} type: {type(value)} not supported."
                "For now, edge attributes other than edge_index are not supported."
            )


def index_select(
    value: TableData,
    index: Tensor,
    dim: int = 0,
) -> TableData:
    r"""Index the :obj:`value` table along dimension :obj:`dim` using
    the entries in :obj:`index`.

    Args:
        value (TableData): The input table.
        index (Tensor): The 1-D tensor containing the indices to select.
        dim (int, optional): The dimension along which to index.
            (default: :obj:`0`)

    Returns:
        TableData: The indexed sub-table.
    """
    index = index.to(torch.int64)

    if isinstance(value, TableData):
        assert dim == 0
        # only slice feature_dict, other attributes
        # like df will be shallow copied.
        return value[index]

    raise ValueError(f"Encountered invalid feature tensor type "
                    f"(got '{type(value)}')")


def filter_node_store_(
    store: NodeStorage,
    out_store: NodeStorage,
    index: Tensor
):
    r"""Filter a node storage to only hold the nodes given by :obj:`index`.

    Args:
        store (NodeStorage): The source node storage.
        out_store (NodeStorage): The output node storage to write into.
        index (Tensor): The 1-D tensor of node indices to keep.
    """
    for key, value in store.items():
        if key == 'num_nodes':
            out_store.num_nodes = index.numel()

        elif store.is_node_attr(key):
            if isinstance(value, TableData):
                out_store[key] = index_select(value, index)
            elif isinstance(value, Tensor):
                # For now, hardcode for `time` tensor in Pkey-fkey graph.
                assert value.dim() == 1, f"Tensor should be 1-D, but {value.dim()} found."
                out_store[key] = value[index]
            else:
                raise NotImplementedError(
                    f"Node attribute type {type(value)} not supported."
                )
