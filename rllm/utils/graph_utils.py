from typing import Union, Tuple, Optional

import torch
from torch import Tensor
import numpy as np
import scipy.sparse as sp

from ._sort import lexsort
from rllm.utils.sparse import is_torch_sparse_tensor, sparse_mx_to_torch_sparse_tensor
import rllm.utils._pyglib

# Filter the csr warning
import warnings

warnings.filterwarnings(
    "ignore",
    message="Sparse CSR tensor support is in beta state.",
    category=UserWarning,
    module=r".*graph_utils",  # discard specific module warning
)


def adj_to_edge_index(adj: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
    r"""Convert a sparse adjacency matrix to an edge index tensor.

    Args:
        adj (Tensor): A sparse adjacency tensor in COO, CSR, or CSC format.

    Returns:
        Tuple[Tensor, Optional[Tensor]]: A tuple :obj:`(edge_index, edge_attr)`
        where :obj:`edge_index` has shape :obj:`[2, num_edges]` and
        :obj:`edge_attr` is :obj:`None` if all edge weights are 1.
    """
    if is_torch_sparse_tensor(adj):
        coo_adj = adj.to_sparse_coo().coalesce()
        s, d, vs = coo_adj.indices()[0], coo_adj.indices()[1], coo_adj.values()
        vs = None if torch.all(vs == 1) else vs
        return torch.stack([s, d]), vs
    else:
        raise TypeError(f"Expect adj to be a SparseTensor, got {type(adj)}.")


def remove_self_loops(adj: Tensor):
    r"""Remove self-loops from the adjacency matrix.

    Args:
        adj (Tensor): The adjacency matrix in sparse or dense format.

    Returns:
        Tensor: The adjacency matrix with self-loops removed.
    """
    shape = adj.shape
    device = adj.device

    if is_torch_sparse_tensor(adj):
        adj = adj.coalesce()
        indices = adj.indices()
        values = adj.values()

        mask = indices[0] != indices[1]
        indices = indices[:, mask]
        values = values[mask]
        return torch.sparse_coo_tensor(indices, values, shape).to(device)

    loop_index = torch.arange(0, shape[0], dtype=torch.long, device=device)
    adj = adj.clone()
    adj[loop_index, loop_index] = 0
    return adj


def gcn_norm(adj: Tensor):
    r"""Normalize the sparse adjacency matrix from the
    `"Semi-supervised Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper.

    .. math::
        \mathbf{\hat{A}} = \mathbf{\hat{D}}^{-1/2} (\mathbf{A} + \mathbf{I})
        \mathbf{\hat{D}}^{-1/2}

    Args:
        adj (Tensor): The sparse adjacency matrix. Supported layouts:
            :obj:`torch.sparse_coo`, :obj:`torch.sparse_csr`,
            :obj:`torch.sparse_csc`.

    Returns:
        Tensor: The symmetrically normalized sparse adjacency matrix.
    """
    shape = adj.shape
    device = adj.device

    if is_torch_sparse_tensor(adj):
        adj = adj.coalesce()
        indices = adj.indices()

        mask = indices[0] != indices[1]
        loop_index = torch.arange(0, shape[0], dtype=torch.long, device=device)
        loop_index = loop_index.unsqueeze(0).repeat(2, 1)

        indices_sl = torch.cat([indices[:, mask], loop_index], dim=1)
        indices = indices[:, mask]

        # D
        adj_data = np.ones([indices.shape[1]], dtype=np.float32)
        adj_sp = sp.csr_matrix((adj_data, (indices[0], indices[1])), shape=shape)

        adj_sl_data = np.ones([indices_sl.shape[1]], dtype=np.float32)
        adj_sl_sp = sp.csr_matrix(
            (adj_sl_data, (indices_sl[0], indices_sl[1])), shape=shape
        )

        # D-1/2
        deg = np.array(adj_sl_sp.sum(axis=1)).flatten()
        deg_sqrt_inv = np.power(deg, -0.5)
        deg_sqrt_inv[deg_sqrt_inv == float("inf")] = 0.0
        deg_sqrt_inv = sp.diags(deg_sqrt_inv)

        # filters
        filters = sp.coo_matrix(deg_sqrt_inv * adj_sp * deg_sqrt_inv)
        return sparse_mx_to_torch_sparse_tensor(filters).to(device)


def sort_edge_index(
    edge_index: Tensor,
    edge_attr: Tensor = None,
    num_nodes: int = None,
    sort_by_row: bool = True,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    r"""Sort the edge index.

    Args:
        edge_index (Tensor): The edge index tensor of shape
            :obj:`[2, num_edges]`.
        edge_attr (Tensor, optional): Edge weights with
            :obj:`size(0) == edge_index.size(1)`. If not :obj:`None`,
            returns :obj:`(edge_index, edge_attr)`.
            (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes. If :obj:`None`,
            inferred from :obj:`edge_index`. (default: :obj:`None`)
        sort_by_row (bool): If set to :obj:`False`, sorts by destination
            node instead of source node. (default: :obj:`True`)

    Example:
        >>> edge_index = torch.tensor([[2, 1, 1, 0],
                                       [1, 2, 0, 1]])
        >>> edge_attr = torch.tensor([[1], [2], [3], [4]])
        >>> sort_edge_index(edge_index)
        tensor([[0, 1, 1, 2],
                [1, 0, 2, 1]])

        >>> sort_edge_index(edge_index, edge_attr)
        (tensor([[0, 1, 1, 2],
                 [1, 0, 2, 1]]),
        tensor([[4],
                [3],
                [2],
                [1]]))
    """
    if num_nodes is None:
        num_nodes = int(edge_index.max().item()) + 1 if edge_index.numel() > 0 else 0
    index = lexsort(
        keys=[
            edge_index[int(sort_by_row)],
            edge_index[1 - int(sort_by_row)],
        ]
    )

    edge_index = edge_index[:, index]

    if edge_attr is not None:
        return edge_index, edge_attr[index]
    return edge_index


def index_to_ptr(index: Tensor, num_nodes: Optional[int] = None) -> Tensor:
    r"""Convert a sorted index tensor to a CSR/CSC pointer tensor.

    Args:
        index (Tensor): The sorted index tensor.
        num_nodes (int, optional): The number of nodes. If :obj:`None`,
            inferred from :obj:`index`. (default: :obj:`None`)

    Example:
        >>> index = torch.tensor([0, 1, 1, 2, 2, 3])
        >>> index_2_ptr(index, 4)
        tensor([0, 1, 3, 5, 6])
    """
    if num_nodes is None:
        num_nodes = int(index.max().item()) + 1 if index.numel() > 0 else 0
    ptr = torch.zeros(num_nodes + 1, dtype=torch.long, device=index.device)
    ptr[1:] = torch.bincount(index, minlength=num_nodes)
    return ptr.cumsum(0)


def _to_csc(
    input: Tensor,
    device: Optional[torch.device] = None,
    num_nodes: Optional[int] = None,
    share_memory: bool = False,
    is_sorted: bool = False,
    src_node_time: Optional[Tensor] = None,
    edge_time: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    r"""Convert an edge index or sparse adjacency matrix to CSC format.

    Args:
        input (Tensor): The input edge index of shape :obj:`[2, num_edges]`
            or a sparse adjacency tensor.
        device (torch.device, optional): The desired device of the returned
            tensors. If :obj:`None`, uses the input device.
            (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes. If :obj:`None`,
            inferred from the edge index. (default: :obj:`None`)
        share_memory (bool): If set to :obj:`True`, moves output tensors to
            shared memory for multi-process data loading.
            (default: :obj:`False`)
        is_sorted (bool): If set to :obj:`True`, skips sorting the edge
            index. (default: :obj:`False`)
        src_node_time (Tensor, optional): Source node timestamps. If not
            :obj:`None`, the edge index is sorted by source node time for
            temporal sampling. (default: :obj:`None`)
        edge_time (Tensor, optional): Edge timestamps. If not :obj:`None`,
            the edge index is sorted by edge time for temporal sampling.
            (default: :obj:`None`)

    Returns:
        Tuple[Tensor, Tensor, Optional[Tensor]]: The column indices,
            row indices, and the permutation index.
    """
    device = input.device if device is None else device
    # sparse adj
    if isinstance(input, Tensor) and input.is_sparse:
        adj = input
        if is_torch_sparse_tensor(adj):
            if src_node_time is not None or edge_time is not None:
                raise NotImplementedError(
                    "Do not support temporal convert for torch sparse tensor."
                )
            csc_t: torch.sparse.Tensor = adj.to_sparse_csc()
            col_ptr = csc_t.ccol_indices()
            row = csc_t.row_indices()
            perm = None
    # edge index
    elif isinstance(input, Tensor) and input.dim() == 2 and input.size(0) == 2:
        row, col = input[0, :], input[1, :]

        if num_nodes is None:
            # If not provided, use max destination node index + 1
            # as the node number.
            # This is used to build col_ptr.
            num_nodes = col.max() + 1

        if not is_sorted:
            if src_node_time is None and edge_time is None:
                perm = torch.argsort(col)
                col = col[perm]
                row = row[perm]
            elif edge_time is not None and src_node_time is None:
                perm = lexsort(keys=[edge_time, col])
                col = col[perm]
                row = row[perm]
            elif src_node_time is not None and edge_time is None:
                perm = lexsort(keys=[src_node_time[row], col])
                col = col[perm]
                row = row[perm]
            else:
                raise NotImplementedError(
                    "Only support one temporal sort for now."
                    "But both `src_node_time` and `edge_time` are not `None`."
                )
        col_ptr = index_to_ptr(col, num_nodes)

    else:
        raise ValueError(
            "No edge found. Edge type should be either `adj` or `edge_index`."
        )

    col_ptr = col_ptr.to(device=device)
    row = row.to(device=device)
    perm = perm.to(device=device) if perm is not None else None

    if not col_ptr.is_cuda and share_memory:
        col_ptr.share_memory_()
        row.share_memory_()
        if perm is not None:
            perm.share_memory_()

    return col_ptr, row, perm


def to_bidirectional(
    row: Tensor,
    col: Tensor,
    rev_row: Tensor,
    rev_col: Tensor,
) -> Tuple[Tensor, Tensor]:
    r"""Merge a directed edge set and its reverse into a deduplicated
    bidirectional edge set.

    Args:
        row (Tensor): Source node indices of the forward edges.
        col (Tensor): Destination node indices of the forward edges.
        rev_row (Tensor): Source node indices of the reverse edges.
        rev_col (Tensor): Destination node indices of the reverse edges.

    Returns:
        Tuple[Tensor, Tensor]: Deduplicated :obj:`(row, col)` tensors
        representing the bidirectional edge set.
    """
    assert row.numel() == col.numel()
    assert rev_row.numel() == rev_col.numel()

    edge_index = row.new_empty(2, row.numel() + rev_row.numel())
    edge_index[0, :row.numel()] = row
    edge_index[1, :row.numel()] = col
    edge_index[0, row.numel():] = rev_col
    edge_index[1, row.numel():] = rev_row

    # Fast path: use PyG's `coalesce` (C++/CUDA-backed via torch-sparse/pyg-lib)
    # to de-duplicate edges. This is significantly faster than `torch.unique`
    # on a 2xE tensor, especially when called repeatedly per edge type/batch.
    if rllm.utils._pyglib.WITH_PYG_LIB:
        try:
            from torch_geometric.utils import coalesce  # type: ignore

            (edge_index, _) = coalesce(
                edge_index,
                None,
                sort_by_row=False,
                reduce='any',
            )
            return edge_index[0], edge_index[1]
        except Exception:
            pass

    # Fallback: pure PyTorch de-duplication.
    edge_index = torch.unique(edge_index, dim=1)
    return edge_index[0], edge_index[1]
