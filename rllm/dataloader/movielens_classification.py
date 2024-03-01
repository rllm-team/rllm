import sys 
import os
from multiprocessing import Pool
current_path = os.path.dirname(__file__)
sys.path.append(current_path + '/../data/')

import pandas as pd
import numpy as np
import torch

import data

rating_range = range(1, 6)
threshold = 2000
net_path = current_path + '/../datasets/rel-movielens1m/classification/'
def load_csv(name):
    return pd.read_csv(net_path + name,
                    sep=',',
                    engine='python',
                    encoding='ISO-8859-1')

def load():
    file_list = [
        '/movies/train.csv',
        '/movies/validation.csv',
        '/movies/test.csv',
        'users.csv',
        'ratings.csv',
    ]
    with Pool(5) as worker:
        train, valid, test, user, rating = worker.map(load_csv, file_list)
    movie_all = pd.concat([test, train, valid])

    genres = movie_all['Genre'].str.get_dummies('|').values
    mid = torch.tensor(movie_all['MovielensID'].values)
    mfeat = np.load(current_path + '/../datasets/embeddings.npy')
    mfeat = torch.Tensor(mfeat)
    label = torch.FloatTensor(genres)#[:, 0: 1]
    
    uid = torch.tensor(user['UserID'].values)
    ufeat = torch.eye(uid.shape[0])

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
        A = torch.spmm(A.transpose(0, 1), A).to_dense()
        hop |= torch.where(A > threshold, 1, 0)
    hop = hop.type(torch.FloatTensor)
    
    return dataset, \
           hop, \
           dataset.x['movie'], \
           dataset.y['movie'], idx_train, idx_val, idx_test