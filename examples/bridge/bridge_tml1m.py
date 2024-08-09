# The BRIDGE method from the "rLLM: Relational Table Learning with LLMs" paper.
# ArXiv: https://arxiv.org/abs/2407.20157

# Datasets  TML1M
# Acc       0.364

import time
import argparse
import os.path as osp
import sys
sys.path.append('../')
sys.path.append('../../')

import torch
import torch.nn.functional as F

import rllm.transforms as T
from rllm.datasets import TML1MDataset
from rllm.nn.models import Bridge
from rllm.transforms import build_homo_graph


parser = argparse.ArgumentParser()
parser.add_argument("--tab_dim", type=int, default=64,
                    help="Tab Transformer categorical embedding dim")
parser.add_argument("--gcn_dropout", type=float, default=0.5,
                    help="Droupout for GCN")

parser.add_argument("--epochs", type=int, default=200,
                    help="Training epochs")
parser.add_argument("--lr", type=float, default=0.001,
                    help="Learning rate")
parser.add_argument("--wd", type=float, default=1e-4,
                    help="Weight decay")
args = parser.parse_args()

# Prepare datasets
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data")
dataset = TML1MDataset(cached_dir=path, force_reload=True)
user_table, movie_table, rating_table, movie_embeddings = dataset.data_list

# We assume it a homogeneous graph,
# so we need to reorder the user and movie id.
ordered_rating = rating_table.df.assign(
    UserID=rating_table.df['UserID']-1,
    MovieID=rating_table.df['MovieID']+len(user_table)-1)

# Making graph
emb_size = movie_embeddings.size(1)
len_user = len(user_table)
len_movie = len(movie_table)
# User embeddings will be further trained
user_embeddings = torch.randn(len_user, emb_size)
x = torch.cat([user_embeddings, movie_embeddings], dim=0)
graph = build_homo_graph(
    df=ordered_rating,
    n_src=len_user,
    n_tgt=len_movie,
    x=x,
    y=user_table.y.long(),
    transform=T.GCNNorm(),
)
graph.user_table = user_table
graph.movie_table = movie_table
graph = graph.to(device)
train_mask, val_mask, test_mask = (
    graph.user_table.train_mask,
    graph.user_table.val_mask,
    graph.user_table.test_mask
)
output_dim = graph.user_table.num_classes


def accuracy_score(preds, truth):
    return (preds == truth).sum(dim=0) / len(truth)


def train_epoch() -> float:
    model.train()
    optimizer.zero_grad()
    logits = model(
        graph.user_table,
        graph.x, graph.adj,
        len_user,
        len_user+len_movie
    )
    loss = F.cross_entropy(
        logits[train_mask].squeeze(), graph.y[train_mask]
    )
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def test_epoch():
    model.eval()
    logits = model(
        graph.user_table,
        graph.x, graph.adj,
        len_user,
        len_user+len_movie
    )
    preds = logits.argmax(dim=1)
    y = graph.y
    train_acc = accuracy_score(preds[train_mask], y[train_mask])
    val_acc = accuracy_score(preds[val_mask], y[val_mask])
    test_acc = accuracy_score(preds[test_mask], y[test_mask])
    return train_acc.item(), val_acc.item(), test_acc.item()


model = Bridge(
    table_hidden_dim=args.tab_dim,
    table_output_dim=emb_size,
    graph_output_dim=output_dim,
    stats_dict=graph.user_table.stats_dict,
    graph_dropout=args.gcn_dropout,
).to(device)


start_time = time.time()
best_val_acc = best_test_acc = 0
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=args.lr,
    weight_decay=args.wd
)
for epoch in range(1, args.epochs + 1):
    train_loss = train_epoch()
    train_acc, val_acc, test_acc = test_epoch()
    print(
        f"Epoch: [{epoch}/{args.epochs}]"
        f"Loss: {train_loss:.4f} train_acc: {train_acc:.4f} "
        f"val_acc: {val_acc:.4f} test_acc: {test_acc:.4f} "
    )
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_test_acc = test_acc
print(
    "Bridge result: "
    f"Best Val acc: {best_val_acc:.4f}, "
    f"Best Test acc: {best_test_acc:.4f}"
)
print(f"Total Time: {time.time() - start_time:.4f}s")
