import sys 
import os
current_path = os.path.dirname(__file__)
sys.path.append(current_path + '/../data')

import scipy.sparse as sp
import numpy as np
import torch

import datatensor

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.torch.sparse_coo_tensor(indices, values, shape)

def load(dataname, device='cpu'):
    import sys
    import pickle as pkl
    import networkx as nx
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open(current_path + "/../datasets/{}/ind.{}.{}".format(dataname, dataname, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file(current_path + "/../datasets/{}/ind.{}.test.index".format(dataname, dataname))
    test_idx_range = np.sort(test_idx_reorder)

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    features = torch.FloatTensor(np.array(features.todense())).float()
    labels = torch.LongTensor(labels)
    
    adj = sparse_mx_to_torch_sparse_tensor(adj).float().coalesce()
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    dataset = datatensor.legacy_init([features], ['v'], 
                [labels], ['v'],
                [adj.indices()], [('e', 'v', 'v')])
    
    dataset.normalize()
    
    dataset.to(device)
    idx_train = idx_train.to(device)
    idx_val = idx_val.to(device)
    idx_test = idx_test.to(device)

    return dataset, dataset.e['e'], dataset.x.to_homo(), dataset.y['v'], idx_train, idx_val, idx_test

# print(load('cora')[0])