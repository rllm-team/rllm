# Datasets  TLF2K   TML1M   TACM12K
# Acc       0.436   0.330   0.234
# Time(s)   280s    723s    286s


import time
import argparse
import os.path as osp
import sys
import random

import torch
import networkx as nx
import dashscope
from langchain_community.llms import Tongyi

sys.path.append("../")
sys.path.append("../..")

from examples.bridge.bridge import build_bridge_model, train_bridge_model
from examples.bridge.utils import data_prepare
from rllm.datasets import TLF2KDataset, TACM12KDataset, TML1MDataset

from utils import annotate

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="tlf2k", choices=["tlf2k", "tml1m", "tacm12k"], help="Dataset")
parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
parser.add_argument("--wd", type=float, default=1e-4, help="Weight decay")
parser.add_argument("--use_cache", type=bool, default=True, help="Whether to use cache")
parser.add_argument("--train_budget", type=int, default=100, help="Annotation budget for training")
parser.add_argument("--val_budget", type=int, default=100, help="Annotation budget for validation")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DASHSCOPE_API_KEY = "your-api-key"
llm = Tongyi(dashscope_api_key=DASHSCOPE_API_KEY, model_kwargs={"api_key": DASHSCOPE_API_KEY, "model": "qwen-max-2025-01-25"}, client=dashscope.Generation)

path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data")
if args.dataset == "tlf2k":
    dataset = TLF2KDataset(cached_dir=path, force_reload=True)
elif args.dataset == "tml1m":
    dataset = TML1MDataset(cached_dir=path, force_reload=True)
elif args.dataset == "tacm12k":
    dataset = TACM12KDataset(cached_dir=path, force_reload=True)
else:
    print("invalid dataset")

target_table, non_table_embeddings, adj, emb_size = data_prepare(dataset, args.dataset, device)

_, val_mask, test_mask = (
    target_table.train_mask.cpu(),
    target_table.val_mask.cpu(),
    target_table.test_mask.cpu(),
)

label_names = sorted(set(map(str, target_table.df[target_table.target_col].tolist())))
select_mask = ~(test_mask | val_mask)
train_num = min(args.train_budget, select_mask.sum())
val_num = min(args.val_budget, val_mask.sum())

start_time = time.time()

target_nodes = list(range(len(target_table.df)))
edges = adj.coalesce().indices().t().tolist()
G = nx.Graph(edges)
target_degrees = [(node, G.degree(node)) for node in target_nodes if select_mask[node]]
target_degrees.sort(key=lambda x: x[1], reverse=True)
top_nodes = [node for node, _ in target_degrees][:train_num]
train_indices = torch.tensor(top_nodes, dtype=torch.long)
train_mask = torch.zeros(target_table.df.shape[0], dtype=torch.bool)
train_mask[train_indices] = True
target_table.train_mask = train_mask

val_indices = torch.nonzero(val_mask).squeeze()
val_indices = val_indices[torch.randperm(len(val_indices))[:val_num]]
val_mask[:] = False
val_mask[val_indices] = True
target_table.val_mask = val_mask

print(f"Using dataset: {args.dataset}")
real_labels = torch.tensor([label_names.index(_) if _ in label_names else random.choice(label_names) for _ in target_table.df[target_table.target_col].astype(str).tolist()])
pseudo_labels = annotate(args.dataset, label_names, target_table, train_mask | val_mask, llm, use_cache=args.use_cache)

y = real_labels.long().to(device)
y[train_mask] = pseudo_labels.long().to(device)[train_mask]
target_table.y = y

model = build_bridge_model(target_table.num_classes, target_table.metadata, emb_size).to(device)
train_bridge_model(model, target_table, non_table_embeddings, adj, args.epochs, args.lr, args.wd, device)

print(f"Annotation and training time: {time.time() - start_time}s")
