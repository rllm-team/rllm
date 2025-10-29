# Datasets  TLF2K   TML1M   TACM12K
# Acc       0.552   0.254   0.293
# Time(s)   3968    5951    5494


import time
import argparse
import os.path as osp
import sys

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
parser.add_argument("--use_cache", type=bool, default=False, help="Whether to use cache")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DASHSCOPE_API_KEY = "sk-dbc98c564b844cd1b15c537f3814ff4d"
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

orig_train_mask, val_mask, test_mask = (
    target_table.train_mask.cpu(),
    target_table.val_mask.cpu(),
    target_table.test_mask.cpu(),
)

label_names = sorted(set(map(str, target_table.df[target_table.target_col].tolist())))
select_mask = ~(test_mask | val_mask)
train_num = min(orig_train_mask.sum(), select_mask.sum())

start_time = time.time()

target_nodes = list(range(len(target_table.df)))
edges = adj.coalesce().indices().t().tolist()
G = nx.Graph(edges)
target_degrees = [(node, G.degree(node)) for node in target_nodes if select_mask[node]]
target_degrees.sort(key=lambda x: x[1], reverse=True)
top_nodes = [node for node, _ in target_degrees][:train_num]
selected_indices = torch.tensor(top_nodes, dtype=torch.long)

train_mask = torch.zeros(target_table.df.shape[0], dtype=torch.bool)
train_mask[selected_indices] = 1
target_table.train_mask = train_mask

print(f"Using dataset: {args.dataset}")
real_labels = torch.tensor([label_names.index(str(_)) if str(_) in label_names else 0 for _ in target_table.df[target_table.target_col].tolist()])
pseudo_labels = annotate(args.dataset, label_names, target_table, train_mask | val_mask, llm, use_cache=args.use_cache)

print("Annotation time:", time.time() - start_time)

y = real_labels.long().to(device)
y[train_mask | val_mask] = pseudo_labels.long().to(device)[train_mask | val_mask]
target_table.y = y

model = build_bridge_model(target_table.num_classes, target_table.metadata, emb_size).to(device)
train_bridge_model(model, target_table, non_table_embeddings, adj, args.epochs, args.lr, args.wd, device)
