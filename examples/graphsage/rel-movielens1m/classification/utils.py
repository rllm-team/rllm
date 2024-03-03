import numpy as np
import torch


# Convert to adjacency list
def adj_matrix_to_list(adj_matrix):
    """
    This function converts adjacency matrices to adjacency lists
    Args:
        adj_matrix (COO Sparse Tensor): The adjacency matrix
        representing the connections between nodes.
    """
    adj_list = {}
    adj_matrix = adj_matrix.to_dense()
    for i in range(adj_matrix.size(0)):
        adj_list[i] = (
            adj_matrix[i] > 0
            ).nonzero(as_tuple=False).squeeze().tolist()
        # Ensure each value is a list, even if there's only one neighbor
        if not isinstance(adj_list[i], list):
            adj_list[i] = [adj_list[i]]
    return adj_list


def sampling(src_nodes, sample_num, neighbor_table):
    """
    Sample a specified number of neighbor nodes based on the source node,
    noting that sampling with replacement is used;
    when the number of neighbor nodes of a certain node is
    less than the sampling number,
    the sampling result will include duplicate nodes.

    Arguments:
        src_nodes {torch.Tensor} -- source nodes
        sample_nums {list of int} -- number of samples for each hop
        neighbor_table {dict} -- mapping from nodes to their neighbors

    Returns:
        torch.Tensor -- result of sampling
    """
    results = []
    for sid in src_nodes:
        # 从节点的邻居中进行有放回地进行采样
        neighbors = neighbor_table[sid.item()]
        if len(neighbors) != 0:
            res = np.random.choice(neighbors, size=(sample_num, ))
        else:
            res = np.array([sid.item()] * sample_num)
        results.append(res)
    return torch.from_numpy(np.asarray(results).flatten())


def multihop_sampling(src_nodes, sample_nums, neighbor_table):
    """Performing multi-hop sampling based on source nodes given

    Arguments:
        src_nodes {torch.Tensor} -- source nodes
        sample_nums {list of int} -- number of samples for each hop
        neighbor_table {dict} -- mapping from nodes to their neighbors

    Returns:
        [list of torch.Tensor] -- result of sampling
    """
    sampling_result = [src_nodes]
    for k, hopk_num in enumerate(sample_nums):
        hopk_result = sampling(sampling_result[k], hopk_num, neighbor_table)
        sampling_result.append(hopk_result)
    return sampling_result
