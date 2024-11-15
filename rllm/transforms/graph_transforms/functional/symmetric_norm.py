import numpy as np
import scipy.sparse as sp
from torch import Tensor

from rllm.utils.sparse import (
    is_torch_sparse_tensor,
    sparse_mx_to_torch_sparse_tensor,
)


def symmetric_norm(adj: Tensor):
    """
    Perform symmetric normalization on the adjacency matrix.

    Args:
        adj (Tensor): the sparse adjacency matrix,
            whose layout could be `torch.sparse_coo`, `torch.sparse_csr`
            and `torch.sparse_csc`.
    """
    shape = adj.shape
    device = adj.device
    if device.type == "cuda":
        adj = adj.to("cpu")

    if is_torch_sparse_tensor(adj):
        adj = adj.coalesce()
        indices = adj.indices()

        # D
        adj_data = np.ones([indices.shape[1]], dtype=np.float32)
        adj_sp = sp.csr_matrix((adj_data, (indices[0], indices[1])), shape=shape)

        # D-1/2
        deg = np.array(adj_sp.sum(axis=1)).flatten()
        deg_sqrt_inv = np.power(deg, -0.5)
        deg_sqrt_inv[deg_sqrt_inv == float("inf")] = 0.0
        deg_sqrt_inv = sp.diags(deg_sqrt_inv)

        # filters
        filters = sp.coo_matrix(deg_sqrt_inv * adj_sp * deg_sqrt_inv)
        return sparse_mx_to_torch_sparse_tensor(filters).to(device)
