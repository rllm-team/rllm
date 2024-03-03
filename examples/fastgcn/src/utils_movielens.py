import pdb

import torch
import numpy as np
import scipy.sparse as sp
import random


def sample_tensor(coo_tensor, idx_list):
    adj_value = coo_tensor._values()
    adj_indice = coo_tensor._indices()
    adj_value_train = adj_value[idx_list]
    adj_indice_train = adj_indice[:, idx_list]
    adj_small = torch.sparse_coo_tensor(
        indices=adj_indice_train, values=adj_value_train,
        size=torch.Size((9923, 9923)))
    return adj_small


def sample_more(coo_tensor, features, labels, idx_list):
    tensor_small = sample_tensor(coo_tensor, idx_list)
    indice_s = tensor_small._indices()
    labels_s = labels[idx_list]
    node_set = set()
    B, N = indice_s.shape
    for i in range(B):
        for j in range(N):
            node_set.add(indice_s[i][j])
    node_list = np.array(list(node_set))
    features_s = features[node_list]

    return tensor_small, features_s, labels_s


def concate_coo(coo_1, coo_2):
    ind1 = coo_1._indices()
    val1 = coo_1._values()
    ind2 = coo_2._indices()
    val2 = coo_2._values()
    # print(f"ind1.shape: {ind1.shape}")
    # print(f"ind2.shape: {ind2.shape}")
    ind_cat = torch.concat((ind1, ind2), dim=1)
    val_cat = torch.concat((val1, val2), dim=0)
    size = torch.Size((coo_1.shape[0] + coo_2.shape[0], coo_1.shape[1]))

    return torch.sparse_coo_tensor(indices=ind_cat, values=val_cat, size=size)


def drop_adj(adj_cat, device):
    adj_drop = torch.zeros_like(adj_cat.to_dense()).to(device)
    for i in range(adj_cat._indices().shape[1]):
        if random.random() < 0.01:
            adj_drop[adj_cat._indices()[0][i], adj_cat._indices()[
                1][i]] = 1
    adj_drop = adj_drop.to_sparse().coalesce()
    return adj_drop


def nontuple_preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    ep = 1e-10
    r_inv = np.power(rowsum + ep, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    # print(adj)
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def nontuple_preprocess_adj(adj):
    adj_normalized = normalize_adj(sp.eye(adj.shape[0]) + adj)
    # adj_normalized = sp.eye(adj.shape[0]) + normalize_adj(adj)
    return adj_normalized.tocsr()


def get_batches(train_ind, train_labels, batch_size=64, shuffle=True):
    """
    Inputs:
        train_ind: np.array
    """
    nums = train_ind.shape[0]
    if shuffle:
        np.random.shuffle(train_ind)
    i = 0
    while i < nums:
        cur_ind = train_ind[i:i + batch_size]
        cur_labels = train_labels[cur_ind]
        yield cur_ind, cur_labels
        i += batch_size


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


if __name__ == '__main__':
    pdb.set_trace()
    adj, features, adj_train, train_features, y_train, y_test, test_index = \
        pdb.set_trace()
