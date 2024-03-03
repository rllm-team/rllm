import sys 
import os
current_path = os.path.dirname(__file__)
sys.path.append(current_path + '/../data')

import pandas as pd
import numpy as np
import torch

import datatensor
import datadf

def _pre_load():
    ddf = datadf.GraphStore()

    net_path = current_path + '/../datasets/rel-movielens1m/regression/'
    df_user = pd.read_csv(net_path + 'users.csv',
                    sep=',',
                    engine='python',
                    encoding='ISO-8859-1')
    ddf.x['user'] = pd.DataFrame(np.eye(len(df_user)))
    umap = datadf._get_id_mapping(df_user['UserID'])

    df_movie = pd.read_csv(net_path + 'movies.csv',
                        sep=',',
                        engine='python',
                        encoding='ISO-8859-1')
    ddf.x['movie'] = pd.DataFrame(np.load(current_path + '/../datasets/embeddings.npy'))
    mmap = datadf._get_id_mapping(df_movie['MovielensID'])
    

    test = pd.read_csv(net_path + 'ratings/test.csv',
                    sep=',',
                    engine='python',
                    encoding='ISO-8859-1')
    train = pd.read_csv(net_path + 'ratings/train.csv',
                        sep=',',
                        engine='python',
                        encoding='ISO-8859-1')
    valid = pd.read_csv(net_path + 'ratings/validation.csv',
                        sep=',',
                        engine='python',
                        encoding='ISO-8859-1')
    rating = pd.concat([test, train, valid])
    edge_index = pd.DataFrame([[umap[_] for _ in rating['UserID'].values], 
                               [mmap[_] for _ in rating['MovieID'].values]])
    edge_weight = rating['Rating']
    ddf.e[('rating', 'user', 'movie')] = (edge_index, edge_weight)
    ddf.y['rating'] = rating['Rating']

    idx_test = torch.LongTensor(test.index)
    idx_train = torch.LongTensor(train.index) + idx_test.shape[0]
    idx_val = torch.LongTensor(valid.index) + idx_test.shape[0] + idx_train.shape[0]

    return ddf, idx_train, idx_val, idx_test

def load(device='cpu'):
    ddf, idx_train, idx_val, idx_test = _pre_load()

    dataset = datatensor.from_datadf(ddf)
    dataset.to(device)
    idx_train = idx_train.to(device)
    idx_val = idx_val.to(device)
    idx_test = idx_test.to(device)

    return dataset, \
           dataset.e.to_homo(), \
           dataset.x.to_homo(), \
           dataset.y['rating'], idx_train, idx_val, idx_test

# print(load()[2].device)