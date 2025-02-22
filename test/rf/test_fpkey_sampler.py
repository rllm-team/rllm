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

sys.path.append("./")
from rllm.datasets import TML1MDataset
from rllm.rf.relationframe import RelationFrame
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

rating_table.table_name = "rating_table"
user_table.table_name = "user_table"
movie_table.table_name = "movie_table"
rf = RelationFrame([user_table, rating_table, movie_table])
my_fpkey_sampler = FPkeySampler(rf, user_table)
for t in rf.tables:
    print(t.table_name)
# print("============")
# sampled_rf, blocks = my_fpkey_sampler.sample([1])
# print(sampled_rf.tables)
# print(blocks)
# print(sampled_rf.tables[0].df, len(sampled_rf.tables[0].df))
