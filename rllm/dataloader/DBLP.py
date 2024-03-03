import sys
import os.path as osp
current_path = osp.dirname(__file__)
sys.path.append(current_path + '/../data')

import scipy.sparse as sp
import numpy as np
import torch
from itertools import product

import datatensor

def load():
    net_path = current_path + '/../datasets/DBLP/'

    v = [None, None, None, None]
    vmeta = ['author', 'paper', 'term', 'conference']
    for i in range(2):
        x = sp.load_npz(osp.join(net_path, f'features_{i}.npz'))
        v[i] = torch.from_numpy(x.todense()).to(torch.float)

    x = np.load(osp.join(net_path, 'features_2.npy'))
    v[2] = torch.from_numpy(x).to(torch.float)

    node_type_idx = np.load(osp.join(net_path, 'node_types.npy'))
    node_type_idx = torch.from_numpy(node_type_idx).to(torch.long)
    v_num = [int((node_type_idx == i).sum()) for i in range(4)]
    v[3] = torch.eye(v_num[3])

    y = np.load(osp.join(net_path, 'labels.npy'))
    y = [torch.from_numpy(y).to(torch.long)]
    ymeta = ['author']

    split = np.load(osp.join(net_path, 'train_val_test_idx.npz'))
    idx_train = torch.from_numpy(split['train_idx']).to(torch.long)
    idx_val = torch.from_numpy(split['val_idx']).to(torch.long)
    idx_test = torch.from_numpy(split['test_idx']).to(torch.long)

    s = {}
    N_a = v_num[0]
    N_p = v_num[1]
    N_t = v_num[2]
    N_c = v_num[3]
    s['author'] = (0, N_a)
    s['paper'] = (N_a, N_a + N_p)
    s['term'] = (N_a + N_p, N_a + N_p + N_t)
    s['conference'] = (N_a + N_p + N_t, N_a + N_p + N_t + N_c)

    # todo: as to_hetero
    edge_index = []
    emeta = []
    A = sp.load_npz(osp.join(net_path, 'adjM.npz'))
    for src, dst in product(vmeta, vmeta):
        A_sub = A[s[src][0]:s[src][1], s[dst][0]:s[dst][1]].tocoo()
        if A_sub.nnz > 0:
            row = torch.from_numpy(A_sub.row).to(torch.long)
            col = torch.from_numpy(A_sub.col).to(torch.long)
            edge_index.append(torch.stack([row, col], dim=0))
            emeta.append((src + '->' + dst, src, dst))

    return datatensor.legacy_init(v, vmeta, y, ymeta, edge_index, emeta)

# print(load())