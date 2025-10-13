import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_test_mask(total_node_number, test_num):
    """
    Split nodes into train, validation and test sets.
    """

    random_indices = torch.randperm(total_node_number)
    test_indices = random_indices[:test_num]
    test_mask = index_to_mask(test_indices, total_node_number)
    return test_mask


def index_to_mask(idx, num_nodes):
    """
    Generate mask based on node indices.
    """
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    mask[idx] = 1
    return mask


def remove_isolated_nodes(mask, adj):
    """
    Remove isolated nodes.
    """
    degrees = torch.sparse.sum(adj, dim=1).to_dense()
    non_isolated_nodes = (degrees != 0).nonzero(as_tuple=True)[0]
    non_isolated_mask = torch.zeros_like(mask)
    non_isolated_mask[non_isolated_nodes] = True
    return mask & non_isolated_mask