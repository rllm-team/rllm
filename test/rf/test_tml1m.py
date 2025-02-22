"""
# Load the data
load from csv files
user_table = Table(...)
movie_table = Table(...)
rating_table = Table(...)

# Construct RelationFrame
relation_frame = RelationFrame(
    tables=[user_table, movie_table, rating_table]
    relations = ...
)

# Dataloader
loader = EntryLoader(relation_frame, batch_size=32, shuffle=True)

# Trainer
for batch in loader:
    out = model(batch)
    ...
"""
import sys
from os import path as osp

import torch
import pandas as pd
import numpy as np

sys.path.append("./")
from rllm.datasets import TML1MDataset
from rllm.data.table_data import TableData
from rllm.rf.relationframe import RelationFrame, Relation
from rllm.rf.sampler.fpkey_sampler import FPkeySampler

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
# emb_size = movie_embeddings.size(1)
# user_size = len(user_table)
# print(emb_size)
# print(user_size)
train_mask, val_mask, test_mask = (
    user_table.train_mask,
    user_table.val_mask,
    user_table.test_mask,
)

print("============")
print(user_table.df)

print("============")
rating_table.table_name = "rating_table"
user_table.table_name = "user_table"
rel = Relation(rating_table, 'UserID', user_table, 'UserID')
print(rel)

print("============")
movie_table.table_name = "movie_table"
rf = RelationFrame([user_table, rating_table, movie_table])
print(rf.relations)

print("============")
print(rf.meta_graph)
# print(rf.meta_graph.edges)
# Directed Graph
print(list(rf.meta_graph.neighbors(user_table))) # []
print(list(rf.meta_graph.neighbors(rating_table))) # [user_table, movie_table]
# Undirected Graph, attr copy
print(list(rf.undirected_meta_graph.neighbors(user_table))) # [rating_table]
assert list(rf.undirected_meta_graph.neighbors(user_table))[0].table_name == "rating_table"
print(rf.undirected_meta_graph.edges[user_table, rating_table]) # {'relation': rating_table.UserID ----> user_table.UserID}

print("============")
my_fpkey_sampler = FPkeySampler(rf, user_table)
print(my_fpkey_sampler.f_p_paths)
assert my_fpkey_sampler.f_p_paths[0] is user_table
assert my_fpkey_sampler.f_p_paths[1] is rating_table
assert my_fpkey_sampler.f_p_paths[2] is movie_table
rel: Relation = rf.undirected_meta_graph.edges[my_fpkey_sampler.f_p_paths[0],
                                      my_fpkey_sampler.f_p_paths[1]]['relation'] # {'relation': rating_table.UserID ----> user_table.UserID}

print("============")
# my_fpkey_sampler.sample([1])
ind = rating_table.fkey_index('UserID')
print(type(ind))
print(ind[np.array([1, 2, 3])])

print(np.array([1]))
sampled_rf, blocks = my_fpkey_sampler.sample([1])
print(sampled_rf.tables)
print(blocks)