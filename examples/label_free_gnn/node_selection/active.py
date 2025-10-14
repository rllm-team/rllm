import networkx as nx
import numpy as np
import torch
from sklearn.metrics.pairwise import euclidean_distances

from .ns_helpers import compute_propagated_features, vertex_cover_by_degree


def vertex_cover_query(budget, edge_index, mask):
    """
    Select nodes using graph covering approach.
    This method was originally proposed by us.
    """
    edges = edge_index.t().tolist()
    G = nx.Graph(edges)
    cover = vertex_cover_by_degree(G, budget, mask)

    selected_indices = torch.tensor(list(cover), dtype=torch.long)

    return selected_indices


def featprop_query(budget, x, edge_index, mask):
    """
    Select nodes using featprop method.
    """

    # Install the package with command: pip install kmedoids
    from kmedoids import fasterpam

    aax_dense = compute_propagated_features(x, edge_index)
    aax_dense[np.where(mask == 0)[0]] = 0
    distmat = euclidean_distances(aax_dense.cpu().numpy())

    km = fasterpam(distmat, budget)
    selected = torch.tensor(np.array(km.medoids, dtype=np.int32))
    select_mask = torch.zeros_like(mask)
    select_mask[selected] = 1
    select_mask = select_mask & mask
    selected_indices = torch.arange(x.shape[0])[select_mask]
    return selected_indices



def random_query(budget, num_nodes, mask):
    """
    Select random nodes.
    """
    selected_indices = torch.arange(0, num_nodes, dtype=torch.long)[mask]
    shuffle_indices = torch.randperm(mask.sum())
    selected_indices = selected_indices[shuffle_indices][:budget]
    return selected_indices
