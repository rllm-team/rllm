import sys
from os import path as osp
import torch

sys.path.append("./")
sys.path.append('../')
sys.path.append('../../')
from rllm.datasets import TML1MDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data")
dataset = TML1MDataset(cached_dir=path, force_reload=True)

# Get the required data
(
    user_table,
    _,
    rating_table,
    movie_embeddings,
) = dataset.data_list
emb_size = movie_embeddings.size(1)
user_size = len(user_table)

target_table = user_table.to(device)
y = user_table.y.long().to(device)
movie_embeddings = movie_embeddings.to(device)