import sys 
import os
from multiprocessing import Pool
current_path = os.path.dirname(__file__)
sys.path.append(current_path + '/../data')

import pandas as pd
import numpy as np
import torch

import data

net_path = current_path + '/../datasets/rel-movielens1m/regression/'
def load_csv(name):
    return pd.read_csv(net_path + name,
                    sep=',',
                    engine='python',
                    encoding='ISO-8859-1')

def load():
    file_list = [
        'movies.csv',
        'users.csv',
        'ratings/test.csv',
        'ratings/train.csv',
        'ratings/validation.csv',
    ]
    with Pool(5) as worker:
        df_m, user, test, train, valid = worker.map(load_csv, file_list)
    mid = torch.tensor(df_m['MovielensID'].values)
    mfeat = torch.eye(len(df_m))
    
    uid = torch.tensor(user['UserID'].values)
    ufeat = torch.eye(uid.shape[0])

    rat = pd.concat([test, train, valid])
    edge_index = torch.Tensor(np.array([rat['UserID'].values, rat['MovieID'].values]))
    # print(edge_index.shape[1])
    label = torch.Tensor(rat['Rating'].values)
    idx_test = torch.LongTensor(test.index)
    idx_train = torch.LongTensor(train.index) + idx_test.shape[0]
    idx_val = torch.LongTensor(valid.index) + idx_test.shape[0] + idx_train.shape[0]

    dataset = data.DataLoader([ufeat, mfeat],
                ['user', 'movie'],
                [label],
                ['rating'],
                [edge_index],
                [('rating', 'user', 'movie')],
                node_index=[uid, mid])

    return dataset, \
           dataset.e.to_homo(), \
           dataset.x.to_homo(), \
           dataset.y['rating'], idx_train, idx_val, idx_test