import sys
import time
from os import path as osp

import torch
from torch import Tensor
import pandas as pd
import numpy as np

sys.path.append("./")
from rllm.datasets import TML1MDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data")
dataset = TML1MDataset(cached_dir=path, force_reload=True)

# Get the required data
(
    user_table,
    movie_table,
    rating_table,
    movie_embeddings,
) = dataset.data_list

def timer(func):
    def wrapper(*args, **kwargs):
        tik = time.perf_counter()
        result = func(*args, **kwargs)
        tok = time.perf_counter()
        print(f"Execution time: {tok - tik} seconds")
        return result
    return wrapper

@timer
def tensor_t():
    tensor_index = Tensor(rating_table.df['UserID'].values)
    res = torch.empty(0)
    for i in [7, 2, 9]:
        cur = torch.nonzero(tensor_index == i).squeeze()
        cur = torch.stack([cur, torch.full_like(cur, i)], dim=1)
        res = torch.cat((res, cur), dim=0)
    return res

@timer
def tensor_t_2():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tensor_index = torch.tensor(rating_table.df['UserID'].values, device=device)
    res = torch.empty(0).to(device)
    for i in range(10000):
        cur = torch.nonzero(tensor_index == i).squeeze()
        cur = torch.stack([cur, torch.full_like(cur, i)], dim=1)
        res = torch.cat((res, cur), dim=0)
    return res

@timer
def pd_t():
    pd_index = rating_table.df['UserID'].values
    res = np.empty((0, 2))
    for i in [7, 2, 9]:
        cur = np.where(pd_index == i)[0]
        cur = np.stack((cur, np.full_like(cur, i)), axis=1)
        res = np.concatenate((res, cur), axis=0)
    return res

@timer
def pd_t_2():
    pd_index = rating_table.df['UserID'].values
    res = np.empty((0, 2))
    for i in range(10000):
        cur = np.where(pd_index == i)[0]
        cur = np.stack((cur, np.full_like(cur, i)), axis=1)
        res = np.concatenate((res, cur), axis=0)
    return res

# print(len(tensor_t_2()))
# print(len(pd_t_2()))
"""
Execution time: 3.685819790000096 seconds
1000209
Execution time: 12.100135104999936 seconds
1000209
"""

print(pd_t()[:10])