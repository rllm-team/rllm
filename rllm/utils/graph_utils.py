from typing import Union, Tuple, Optional

import torch
from torch import Tensor
import numpy as np
import scipy.sparse as sp

from rllm.utils._sort import lexsort
from rllm.utils.sparse import is_torch_sparse_tensor, sparse_mx_to_torch_sparse_tensor

# Filter the csr warning
import warnings

warnings.filterwarnings(
    "ignore",
    message="Sparse CSR tensor support is in beta state.",
    category=UserWarning,
    module=r".*graph_utils",  # 只屏蔽特定模块的警告
)


def adj2edge_index(adj: Tensor) -> Union[Tensor, Optional[Tensor]]:
    r"""Transfer sparse adj to edge_index."""
    if adj.is_sparse:
        coo_adj = adj.to_sparse_coo().coalesce()
        s, d, vs = coo_adj.indices()[0], coo_adj.indices()[1], coo_adj.values()
        vs = None if torch.all(vs == 1) else vs
        return torch.stack([s, d]), vs
    else:
        raise TypeError(f"Expect adj to be a SparseTensor, got {type(adj)}.")


def remove_self_loops(adj: Tensor):
    r"""Remove self-loops from the adjacency matrix.

    Args:
        adj (Tensor): the adjacency matrix.
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


def add_remaining_self_loops(adj: Tensor, fill_value=1.0):
    r"""Add self-loops into the adjacency matrix.

    .. math::
        $\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}$

    Args:
        adj (Tensor): the adjacency matrix.
        fill_value (Any): values to be filled in the self-loops,
            the default values is 1.
    """
    shape = adj.shape
    device = adj.device

    if is_torch_sparse_tensor(adj):
        adj = adj.coalesce()
        indices = adj.indices()
        values = adj.values()

        mask = indices[0] != indices[1]

        loop_index = torch.arange(0, shape[0], dtype=torch.long, device=device)
        loop_index = loop_index.unsqueeze(0).repeat(2, 1)

        indices = torch.cat([indices[:, mask], loop_index], dim=1)
        fill_values = (
            torch.ones_like(loop_index, dtype=values.dtype, device=device) * fill_value
        )
        values = torch.cat([values[mask], fill_values], dim=0)
        return torch.sparse_coo_tensor(indices, values, shape).to(device)

    loop_index = torch.arange(0, shape[0], dtype=torch.long, device=device)
    adj = adj.clone()
    adj[loop_index, loop_index] = fill_value
    return adj


def construct_graph(
    edge_index: Tensor, N: int, edge_attr: Tensor = None, remove_self: bool = True
):
    r"""Convert a edge index matrix to a sparse adjacency matrix.

    Args:
        edge_index (Tensor): the edge index.
        N (int): numbers of the nodes.
        edge_attr (Tensor): values of the non-zero elements.
        removed_self (bool): If set to `False`, not remove self-loops
            in the adjecency matrix.
    """
    device = edge_index.device
    edge_index = edge_index.cpu()

    if remove_self:
        edge_index = remove_self_loops(edge_index=edge_index)

    if edge_attr is None:
        edge_attr = np.ones([edge_index.shape[1]], dtype=np.float32)

    adj_sp = sp.csr_matrix((edge_attr, (edge_index[0], edge_index[1])), shape=[N, N])
    return sparse_mx_to_torch_sparse_tensor(adj_sp).to(device)


def gcn_norm(adj: Tensor):
    r"""Normalize the sparse adjacency matrix from the `"Semi-supervised
    Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`__ .

    .. math::
        \mathbf{\hat{A}} = \mathbf{\hat{D}}^{-1/2} (\mathbf{A} + \mathbf{I})
        \mathbf{\hat{D}}^{-1/2}

    Args:
        adj (Tensor): the sparse adjacency matrix,
            whose layout could be `torch.sparse_coo`, `torch.sparse_csr`
            and `torch.sparse_csc`.
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
        edge_index (Tensor): The edge index tensor.
        edge_attr (Tensor, optional): Edge weights, should be a tensor with
            `size(0) == edge_index.size(1)`. If not None, return
            `(edge_index, edge_attr)`, else return `edge_index` only.
            (default: `None`)
        num_nodes (int, optional): The number of nodes.
            If None, infer from edge_index.
            (default: `None`)
        sort_by_row (bool): If set to `False`, will sort `edge_index`
            column-wise/by destination node.
            (default: `True`)

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


def index2ptr(index: Tensor, num_nodes: Optional[int] = None) -> Tensor:
    r"""Convert the sorted index tensor to the pointer tensor.

    Args:
        index (Tensor): The index tensor.
        num_nodes (int, optional): The number of nodes.
            (default: `None`)

    Example:
        >>> index = torch.tensor([0, 1, 1, 2, 2, 3])
        >>> index2ptr(index, 4)
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
    r"""Convert the input edge_index or adj to a CSC format.

    Args:
        input (Tensor): The input edge_index or adj.
        device (torch.device, optional): The desired device of the
            returned tensors. If None, use the current device.
            (default: `None`)
        num_nodes (int, optional): The number of nodes.
            If None, infer from edge_index.
            (default: `None`)
        share_memory (bool, optional): If set to `True`, will share memory
            among returned tensors. This can accelerate process when using
            multiple processes.
            (default: `False`)
        is_sorted (bool, optional): If set to `True`, will not sort the
            edge index.
            (default: `False`)
        src_node_time (Tensor, optional): The source node time.
            If not None, will sort the edge index by `src_node_time`.
            (default: `None`)
        edge_time (Union[str, Tensor], optional): The edge time attribute.
            If not None, will sort the edge index by `edge_time_attr`.
            (default: `None`)

    Returns:
        Tuple[Tensor, Tensor, Optional[Tensor]]: The column indices,
            row indices, and the permutation index.
    """
    device = input.device if device is None else device
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
    elif isinstance(input, Tensor) and input.dim() == 2 and input.size(0) == 2:
        row, col = input[0, :], input[1, :]

        if num_nodes is None:
            num_nodes = max(row.max(), col.max()) + 1

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
                perm = lexsort(keys=[src_node_time, col])
                col = col[perm]
                row = row[perm]
            else:
                raise NotImplementedError(
                    "Only support one temporal sort for now."
                    "But both `src_node_time` and `edge_time` are not `None`."
                )
        col_ptr = index2ptr(col, num_nodes)

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
