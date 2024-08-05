import numpy as np
import torch
from sklearn.cluster import KMeans


def generate_test_mask(total_node_number):
    """
    Split nodes into train, validation and test sets.
    """
    test_num = int(total_node_number * 0.2)

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


def vertex_cover_by_degree(G, b, mask):
    """
    Recursively select the node with the highest degree and remove all of its neighbours
    """
    G_copy = G.copy()
    cover = []
    counter = 0
    while G_copy.edges() and counter < b:
        max_degree_node = max(G_copy.degree, key=lambda x: x[1])[0]
        neighbors = list(G_copy.neighbors(max_degree_node))
        G_copy.remove_node(max_degree_node)
        if mask[max_degree_node]:
            G_copy.remove_nodes_from(neighbors)
            cover.append(max_degree_node)
            counter += 1


    return cover


def get_entropy(label_tensor):
    """
    Compute entropy of label tensor.
    """
    unique_labels, counts = label_tensor.unique(return_counts=True)
    probabilities = counts.float() / label_tensor.size(0)
    entropy = -torch.sum(probabilities * torch.log2(probabilities))
    return entropy


def get_entropy_change(labels):
    """
    Compute entropy changes after removal of each label.
    """
    label_list = labels.tolist()
    original_entropy = get_entropy(labels)
    mask = torch.ones_like(labels, dtype=torch.bool)

    changes = []
    for i, y in enumerate(labels):
        if label_list.count(y.item()) == 1:
            changes.append(-np.inf)
            continue
        temp_train_mask = mask.clone()
        temp_train_mask[i] = 0
        v_labels = labels[temp_train_mask]
        new_entropy = get_entropy(v_labels)
        diff = original_entropy - new_entropy
        changes.append(diff)
    return torch.tensor(changes)


def compute_propagated_features(x, edge_index):
    """
    Compute node values after normalization and feature propagation.
    """
    num_nodes = x.shape[0]
    edge_weight = torch.ones(edge_index.size(1), dtype=torch.float32, device=edge_index.device)
    row, col = edge_index[0], edge_index[1]
    deg = torch.zeros(num_nodes)
    deg.index_add_(0, row, edge_weight)
    deg_inv = 1.0 / deg
    deg_inv.masked_fill_(deg_inv == float('inf'), 0)
    edge_weight = deg_inv[row] * edge_weight
    adj = torch.sparse_coo_tensor(edge_index, edge_weight, size=(num_nodes, num_nodes))
    adj_matrix2 = adj.matmul(adj)
    aax = adj_matrix2.matmul(x)
    x = aax.to_dense()
    return x


def get_density(data):
    """
    Compute density which is defined as the distance to the closest cluster center.
    """
    x = data.x
    n_clusters = data.num_classes
    kmeans = KMeans(n_clusters=n_clusters)
    aax = compute_propagated_features(x, data.adj.coalesce().indices())

    kmeans.fit(aax)

    centers = kmeans.cluster_centers_
    centers = torch.tensor(centers, dtype=x.dtype, device=x.device)

    label = kmeans.predict(aax)
    label = torch.tensor(label)
    centers = centers[label]
    dist_map = torch.linalg.norm(aax - centers, dim=1)

    dist_map = dist_map.clone().detach().to(dtype=x.dtype, device=x.device)

    density = 1 / (1 + dist_map)

    return density


def remove_isolated_nodes(mask, adj):
    """
    Remove isolated nodes.
    """
    degrees = torch.sparse.sum(adj, dim=1).to_dense()
    non_isolated_nodes = (degrees != 0).nonzero(as_tuple=True)[0]
    non_isolated_mask = torch.zeros_like(mask)
    non_isolated_mask[non_isolated_nodes] = True
    return mask & non_isolated_mask
