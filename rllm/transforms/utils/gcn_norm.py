import numpy as np
import scipy.sparse as sp

import torch
from torch import Tensor

from rllm.utils.sparse import (
    is_torch_sparse_tensor,
    sparse_mx_to_torch_sparse_tensor
)


def gcn_norm(adj: Tensor, loop: bool):
    r"""Normalize the sparse adjacency matrix from the "Semi-supervised
    Classification with Graph Convolutional
    Networks" <https://arxiv.org/abs/1609.02907>.

    .. math::
        $\mathbf{\hat{A}} = \mathbf{\hat{D}}^{-1/2} (\mathbf{A} + \mathbf{I})
        \mathbf{\hat{D}}^{-1/2}$

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
        adj_sp = sp.csr_matrix(
            (adj_data, (indices[0], indices[1])),
            shape=shape
        )

        adj_sl_data = np.ones([indices_sl.shape[1]], dtype=np.float32)
        adj_sl_sp = sp.csr_matrix(
            (adj_sl_data, (indices_sl[0], indices_sl[1])),
            shape=shape
        )

        # D-1/2
        deg = np.array(adj_sl_sp.sum(axis=1)).flatten()
        deg_sqrt_inv = np.power(deg, -0.5)
        deg_sqrt_inv[deg_sqrt_inv == float('inf')] = 0.0
        deg_sqrt_inv = sp.diags(deg_sqrt_inv)

        # filters
        if loop:
            filters = sp.coo_matrix(deg_sqrt_inv * adj_sl_sp * deg_sqrt_inv)
            return sparse_mx_to_torch_sparse_tensor(filters).to(device)
        filters = sp.coo_matrix(deg_sqrt_inv * adj_sp * deg_sqrt_inv)
        return sparse_mx_to_torch_sparse_tensor(filters).to(device)
        
