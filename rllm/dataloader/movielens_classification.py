import sys 
import os
current_path = os.path.dirname(__file__)
sys.path.append(current_path + '/../data/')

import pandas as pd
import numpy as np
import torch

import datatensor
import datadf

rating_range = range(1, 6)
threshold = 2000

def _pre_load():
    ddf = datadf.GraphStore()

    net_path = current_path + '/../datasets/rel-movielens1m/classification/'
    train = pd.read_csv(net_path + '/movies/train.csv',
                        sep=',',
                        engine='python',
                        encoding='ISO-8859-1')
    valid = pd.read_csv(net_path + '/movies/validation.csv',
                        sep=',',
                        engine='python',
                        encoding='ISO-8859-1')
    test = pd.read_csv(net_path + '/movies/test.csv',
                       sep=',',
                       engine='python',
                       encoding='ISO-8859-1')
    movie_all = pd.concat([test, train, valid])

    ddf.x['movie'] = pd.DataFrame(np.load(current_path + '/../datasets/embeddings.npy'))
    ddf.y['movie'] = movie_all['Genre'].str.get_dummies('|')
    mmap = datadf._get_id_mapping(movie_all['MovielensID'])
    
    user = pd.read_csv(net_path + 'users.csv',
                       sep=',',
                       engine='python',
                       encoding='ISO-8859-1')
    ddf.x['user'] = pd.DataFrame(np.eye(len(user)))
    umap = datadf._get_id_mapping(user['UserID'])


    rating = pd.read_csv(net_path + 'ratings.csv',
                         sep=',',
                         engine='python',
                         encoding='ISO-8859-1')
    edge_index = pd.DataFrame([[umap[_] for _ in rating['UserID'].values], 
                               [mmap[_] for _ in rating['MovieID'].values]])
    edge_weight = rating['Rating']
    ddf.e[('rating', 'user', 'movie')] = (edge_index, edge_weight)
    
    trainid = train['MovielensID'].values
    validid = valid['MovielensID'].values
    testid = test['MovielensID'].values
    idx_train = torch.LongTensor([mmap[i] for i in trainid])
    idx_val = torch.LongTensor([mmap[i] for i in validid])
    idx_test = torch.LongTensor([mmap[i] for i in testid])
    
    return ddf, idx_train, idx_val, idx_test


def load(device='cpu'):
    ddf, idx_train, idx_val, idx_test = _pre_load()

    dataset = datatensor.from_datadf(ddf)
    dataset.to(device)
    idx_train = idx_train.to(device)
    idx_val = idx_val.to(device)
    idx_test = idx_test.to(device)

    adj = dataset.e['rating']
    adj_i, adj_v = adj.indices(), adj.values()
    hop = torch.zeros((dataset.node_count('movie'), dataset.node_count('movie'))).type(torch.LongTensor)
    hop = hop.to(device)
    for i in rating_range:
        idx = torch.where(adj_v == i, True, False)
        A = torch.sparse_coo_tensor(adj_i[:, idx], adj_v[idx], adj.shape)
        A = torch.spmm(A.transpose(0, 1), A).to_dense()
        hop |= torch.where(A > threshold, 1, 0)
    hop = hop.type(torch.FloatTensor).to(device)

    return dataset, \
           hop, \
           dataset.x['movie'], \
           dataset.y['movie'], idx_train, idx_val, idx_test

# print(load()[2].device)