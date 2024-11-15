import torch
from torch import Tensor

from rllm.utils.sparse import is_torch_sparse_tensor


def add_remaining_self_loops(adj: Tensor, fill_value=1.0):
    r"""Add self-loops into the adjacency matrix.

    .. math::
        \mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}

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
        fill_values = (
            torch.ones_like(loop_index, dtype=values.dtype, device=device) * fill_value
        )

        loop_index = loop_index.unsqueeze(0).repeat(2, 1)
        indices = torch.cat([indices[:, mask], loop_index], dim=1)
        values = torch.cat([values[mask], fill_values], dim=0)
        return torch.sparse_coo_tensor(indices, values, shape).to(device)

    loop_index = torch.arange(0, shape[0], dtype=torch.long, device=device)
    adj = adj.clone()
    adj[loop_index, loop_index] = fill_value
    return adj
