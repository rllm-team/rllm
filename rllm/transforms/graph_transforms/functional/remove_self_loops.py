
import torch
from torch import Tensor

from rllm.utils.sparse import is_torch_sparse_tensor


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
