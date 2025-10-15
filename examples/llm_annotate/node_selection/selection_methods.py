import networkx as nx
import torch


def degree_query(budget, edge_index, mask):
    """
    Select nodes based on node degrees.
    This method selects the top-degree nodes under the given budget.
    """
    edges = edge_index.t().tolist()
    G = nx.Graph(edges)

    degree_list = [(node, deg) for node, deg in G.degree() if mask[node]]
    degree_list.sort(key=lambda x: x[1], reverse=True)

    top_nodes = [node for node, _ in degree_list[:budget]]
    selected_indices = torch.tensor(top_nodes, dtype=torch.long)

    return selected_indices


def random_query(budget, num_nodes, mask):
    """
    Select random nodes.
    """
    selected_indices = torch.arange(0, num_nodes, dtype=torch.long)[mask]
    shuffle_indices = torch.randperm(mask.sum())
    selected_indices = selected_indices[shuffle_indices][:budget]
    return selected_indices