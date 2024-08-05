import torch
from torch import Tensor


def is_undirected(adj: Tensor):
    """Checks if the given adjacency matrix represents an undirected graph.
    Args:
        adj (Tensor): The adjacency matrix in sparse format.

    Returns:
        bool: True if the graph is undirected, False otherwise.
    """
    M, N = adj.shape
    if M != N:
        return False

    # Ensure the adjacency matrix is in coalesced format
    adj = adj.coalesce()
    edge_index = adj.indices()
    values = adj.values()

    # Sort the source and target indices
    src1, indices1 = torch.sort(edge_index[0])
    tgt1 = edge_index[1][indices1]

    src2, indices2 = torch.sort(edge_index[1])
    tgt2 = edge_index[0][indices2]

    # Check if the sorted source and target indices match
    if not torch.equal(src1, src2):
        return False
    if not torch.equal(tgt1, tgt2):
        return False

    # Check if the edge values match
    if not torch.equal(values[indices1], values[indices2]):
        return False

    return True


def to_undirected(adj: Tensor):
    """Converts the given adjacency matrix to anundirected
    graph representation.

    Args:
        adj (Tensor): The adjacency matrix in sparse format.

    Returns:
        Tensor: The undirected adjacency matrix in sparse format.
    """
    # Determine the size of the adjacency matrix
    N = max(adj.shape[0], adj.shape[1])

    # Ensure the adjacency matrix is in coalesced format
    adj = adj.coalesce()
    row, col = adj.indices()
    values = adj.values()

    # Concatenate the indices and values to form an undirected graph
    row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)
    edge_index = torch.stack([row, col], dim=0)
    values = torch.cat([values, values], dim=0)

    # Create and return the undirected adjacency matrix
    return torch.sparse_coo_tensor(edge_index, values, (N, N)).coalesce()
