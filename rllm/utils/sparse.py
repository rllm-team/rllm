import numpy as np
import torch
from torch import Tensor


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    r"""Convert a scipy sparse matrix to a :class:`torch.sparse.Tensor`.

    Args:
        sparse_mx (scipy.sparse.spmatrix): The input scipy sparse matrix.

    Returns:
        Tensor: A sparse COO tensor with :obj:`float32` values.
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)


def is_torch_sparse_tensor(src):
    r"""Return :obj:`True` if the input is a :class:`torch.sparse.Tensor`.

    Args:
        src (Any): The object to check.

    Returns:
        bool: :obj:`True` if :obj:`src` is a sparse tensor, :obj:`False`
        otherwise.
    """
    sparse_types = [torch.sparse_coo, torch.sparse_csr, torch.sparse_csc]
    if isinstance(src, torch.Tensor):
        return src.layout in sparse_types
    return False


def get_indices(adj: Tensor):
    r"""Get indices of non-zero elements from an adjacency matrix.

    Args:
        adj (Tensor): the adjacency matrix.
    Returns:
        indices (Tensor): indices of non-zero elements.
    """
    if is_torch_sparse_tensor(adj):
        indices = adj.coalesce().indices()
    else:
        indices = adj.nonzero().t()
    return indices


def set_values(adj: Tensor, values: Tensor) -> Tensor:
    r"""Replace the values of a sparse tensor while keeping its indices.

    Args:
        adj (Tensor): The sparse tensor whose values are to be replaced.
            Must be in COO, CSR, or CSC format.
        values (Tensor): The new values. Must match the number of non-zero
            elements in :obj:`adj`.

    Returns:
        Tensor: A new sparse tensor with the same sparsity pattern as
        :obj:`adj` but with the updated :obj:`values`.
    """
    if values.dim() > 1:
        size = adj.size() + values.size()[1:]
    else:
        size = adj.size()

    if adj.layout == torch.sparse_coo:
        return torch.sparse_coo_tensor(
            adj.indices(), values, size, device=adj.device
        )
    elif adj.layout == torch.sparse_csr:
        return torch.sparse_csr_tensor(
            adj.crow_indices(), adj.col_indices(), values, size, device=adj.device
        )
    elif adj.layout == torch.sparse_csc:
        return torch.sparse_csc_tensor(
            adj.ccol_indices(), adj.row_indices(), values, size, device=adj.device
        )
    else:
        raise ValueError(f"Unsupported sparse tensor layout: {adj.layout}")
