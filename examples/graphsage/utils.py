import numpy as np
import torch


# Convert to adjacency list
def adj_matrix_to_list(adj_matrix, node_index_movie=None, label_mat=None, task="default"):
    """
    This function converts adjacency matrices to adjacency lists and
    label matrices to label lists for a given set of nodes,
    typically representing movies.
    Args:
        adj_matrix (COO Sparse Tensor): The adjacency matrix representing
        the connections between nodes.
        node_index_movie (Iterable): An iterable of indices representing
        specific nodes (e.g., movies), useless in cora and
        movie-classfication
        label_mat (COO Sparse Tensor): Matrix containing labels, with
        the same shape and indices as adj_matrix, useless in cora and
        movie-classfication
        task (string):The task that this function used, default as cora and
        movie-classfication
    """
    if task == "movie-reg":
        adj_list = {}
        label_list = {}
        adj_matrix = adj_matrix.transpose(0, 1)
        label_mat = label_mat.transpose(0, 1)
        for i in node_index_movie:
            i = i.item()
            adj_list[i] = adj_matrix[i].coalesce().indices().squeeze().numpy()
            label_list[i] = label_mat[i].coalesce().values().squeeze().numpy()
            if len(adj_list[i].shape) == 0:
                adj_list[i] = np.expand_dims(adj_list[i], 0)
            if len(label_list[i].shape) == 0:
                label_list[i] = np.expand_dims(label_list[i], 0)
        return adj_list, label_list
    else:
        adj_list = {}
        adj_matrix = adj_matrix.to_dense()
        for i in range(adj_matrix.size(0)):
            adj_list[i] = (adj_matrix[i] > 0).nonzero(as_tuple=False)\
                .squeeze().tolist()
            # Ensure each value is a list, even if there's only one neighbor
            if not isinstance(adj_list[i], list):
                adj_list[i] = [adj_list[i]]
        return adj_list


def sampling(src_nodes, sample_num, neighbor_table, task, label_table=None):
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
        task {str} -- the task that this function used
        label_table {COO Sparse Tensor} --  Matrix containing labels, with
        the same shape and indices as adj_matrix, useless in cora and
        movie-classfication

    Returns:
        torch.Tensor -- result of sampling
    """
    if task == "movie-reg":
        results_neighbor = []
        results_label = []
        for sid in src_nodes:
            sid = sid.item()
            try:
                num_neighbors = len(neighbor_table[sid])
            except KeyError:
                num_neighbors = 0
            if num_neighbors == 0:
                res_nei = np.array([sid] * sample_num)
                res_label = np.array([3] * sample_num)
            else:
                sample_indices = np.random.randint(
                    0, num_neighbors, size=(sample_num, )
                    )
                res_nei = neighbor_table[sid][sample_indices]
                res_label = label_table[sid][sample_indices]
            results_neighbor.append(res_nei)
            results_label.append(res_label)
        return torch.from_numpy(np.asarray(results_neighbor).flatten()), \
            torch.from_numpy(np.asarray(results_label).flatten())
    elif task == "movie-cla":
        results = []
        for sid in src_nodes:
            neighbors = neighbor_table[sid.item()]
            if len(neighbors) != 0:
                res = np.random.choice(neighbors, size=(sample_num, ))
            else:
                res = np.array([sid.item()] * sample_num)
            results.append(res)
        return torch.from_numpy(np.asarray(results).flatten())
    else:
        results = []
        for sid in src_nodes:
            res = np.random.choice(
                neighbor_table[sid.item()],
                size=(sample_num, ))
            results.append(res)
        return torch.from_numpy(np.asarray(results).flatten())


def multihop_sampling(src_nodes, sample_nums, neighbor_table, task, label_table=None):
    """Performing multi-hop sampling based on source nodes given

    Arguments:
        src_nodes {torch.Tensor} -- source nodes
        sample_nums {list of int} -- number of samples for each hop
        neighbor_table {dict} -- mapping from nodes to their neighbors
        task {str} -- the task that this function used
        label_table {COO Sparse Tensor} --  Matrix containing labels, with
        the same shape and indices as adj_matrix, useless in cora and
        movie-classfication

    Returns:
        [list of torch.Tensor] -- result of sampling
    """
    if task == "movie-reg":
        sampling_result = [src_nodes]
        label_return = None
        for k, hopk_num in enumerate(sample_nums):
            hopk_result, hopk_label = sampling(
                sampling_result[k],
                hopk_num,
                neighbor_table,
                task,
                label_table)
            if k == 0:
                label_return = hopk_label
            sampling_result.append(hopk_result)
        return sampling_result, label_return
    else:
        sampling_result = [src_nodes]
        for k, hopk_num in enumerate(sample_nums):
            hopk_result = sampling(
                sampling_result[k],
                hopk_num,
                neighbor_table,
                task)
            sampling_result.append(hopk_result)
        return sampling_result
