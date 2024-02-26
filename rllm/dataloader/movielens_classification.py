import sys 
import os
current_path = os.path.dirname(__file__)
sys.path.append(current_path + '/../data/')

import pandas as pd
import numpy as np
import torch

import data

rating_range = range(1, 6)
threshold = 2000

def load():
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

    genres = movie_all['Genre'].str.get_dummies('|').values
    mid = torch.tensor(movie_all['MovielensID'].values)
    mfeat = np.load(current_path + '/../datasets/embeddings.npy')
    mfeat = torch.Tensor(mfeat)
    label = torch.FloatTensor(genres)#[:, 0: 1]
    
    user = pd.read_csv(net_path + 'users.csv',
                       sep=',',
                       engine='python',
                       encoding='ISO-8859-1')
    uid = torch.tensor(user['UserID'].values)
    ufeat = torch.eye(uid.shape[0])

    rating = pd.read_csv(net_path + 'ratings.csv',
                         sep=',',
                         engine='python',
                         encoding='ISO-8859-1')
    edge_index = torch.Tensor(np.array([rating['UserID'].values, rating['MovieID'].values]))
    edge_weight = torch.Tensor(rating['Rating'].values)

    dataset = data.DataLoader([ufeat, mfeat],
                    ['user', 'movie'],
                    [label],
                    ['movie'],
                    [edge_index],
                    [('rating', 'user', 'movie')],
                    node_index=[uid, mid],
                    edge_weight=[edge_weight])
    
    vmap = dataset.x.vmap['movie']
    trainid = train['MovielensID'].values
    validid = valid['MovielensID'].values
    testid = test['MovielensID'].values
    idx_train = torch.LongTensor([vmap[i] for i in trainid])
    idx_val = torch.LongTensor([vmap[i] for i in validid])
    idx_test = torch.LongTensor([vmap[i] for i in testid])

    adj = dataset.e['rating']
    adj_i, adj_v = adj.indices(), adj.values()
    hop = torch.zeros((dataset.v_num['movie'], dataset.v_num['movie'])).type(torch.LongTensor)
    for i in rating_range:
        idx = torch.where(adj_v == i, True, False)
        A = torch.sparse_coo_tensor(adj_i[:, idx], adj_v[idx], adj.shape)
        A = torch.spmm(A.T, A).to_dense()
        hop |= torch.where(A > threshold, 1, 0)
    hop = hop.type(torch.FloatTensor)
    
    return dataset, \
           hop, \
           dataset.x['movie'], \
           dataset.y['movie'], idx_train, idx_val, idx_test