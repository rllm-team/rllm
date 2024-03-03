
import sys 
sys.path.append("../../rllm/data")

import numpy as np
from scipy.io import loadmat
import torch

import datatensor

def scicsc_to_torch_sparse(csc):
    adj = csc.tocoo()

    values = adj.data
    indices = np.vstack((adj.row, adj.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = adj.shape

    return torch.sparse_coo_tensor(i, v, torch.Size(shape)).coalesce()

def load():
    net_path = '../../rllm/datasets/small_alibaba_1_10/small_alibaba_1_10.mat'

    mat = loadmat(net_path)
    # e0, e1, e2, e3
    A_mat = mat['train'] + mat['valid'] + mat['test']
    A_torch = [scicsc_to_torch_sparse(A_mat[_][0]) for _ in range(4)]
    
    # feature
    feature = scicsc_to_torch_sparse(mat['full_feature']).to_dense()

    # label (partial)
    label = scicsc_to_torch_sparse(mat['label'])
    idx_train = torch.LongTensor(mat['train_idx'].ravel().astype(np.int16)) - 1
    idx_val = torch.LongTensor(mat['valid_idx'].ravel().astype(np.int16)) - 1
    idx_test = torch.LongTensor(mat['test_idx'].ravel().astype(np.int16)) - 1
    
    dataset = datatensor.legacy_init([feature], ['v'], 
                [label], ['v'],
                A_torch, [(0, 'v', 'v'), (1, 'v', 'v'), (2, 'v', 'v'), (3, 'v', 'v')])

    return dataset, dataset.e, dataset.x.to_homo(), dataset.y['v'], idx_train, idx_val, idx_test