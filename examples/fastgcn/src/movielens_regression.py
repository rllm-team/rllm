import data
import torch
import numpy as np
import pandas as pd
import sys
import os
current_path = os.path.dirname(__file__)
current_path = current_path + "/../../../rllm"
sys.path.append(current_path + '/../data')


def load():
    net_path = current_path + '/datasets/rel-movielens1m/regression/'
    df_m = pd.read_csv(net_path + 'movies.csv',
                       sep=',',
                       engine='python',
                       encoding='ISO-8859-1')
    mid = torch.tensor(df_m['MovielensID'].values)
    mfeat = torch.eye(len(df_m))

    user = pd.read_csv(net_path + 'users.csv',
                       sep=',',
                       engine='python',
                       encoding='ISO-8859-1')
    uid = torch.tensor(user['UserID'].values)
    ufeat = torch.eye(uid.shape[0])

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
    rat = pd.concat([test, train, valid])
    edge_index = torch.Tensor(
        np.array([rat['UserID'].values, rat['MovieID'].values]))
    # print(edge_index.shape[1])
    label = torch.Tensor(rat['Rating'].values)
    idx_test = torch.LongTensor(test.index)
    idx_train = torch.LongTensor(train.index) + idx_test.shape[0]
    idx_val = torch.LongTensor(valid.index) + \
        idx_test.shape[0] + idx_train.shape[0]

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
