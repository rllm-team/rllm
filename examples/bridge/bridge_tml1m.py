# The BRIDGE method from the "rLLM: Relational Table Learning with LLMs" paper.
# ArXiv: https://arxiv.org/abs/2407.20157

# Datasets  TML1M
# Acc       0.428

import time
import argparse
import os.path as osp
import sys

sys.path.append("./")
sys.path.append("../")
sys.path.append("../../")

import torch
import torch.nn.functional as F

import rllm.transforms.graph_transforms as GT
from rllm.datasets import TML1MDataset
from utils import build_homo_data, build_homo_graph, GraphEncoder, TableEncoder


parser = argparse.ArgumentParser()
parser.add_argument("--gcn_dropout", type=float, default=0.5, help="Dropout for GCN")
parser.add_argument("--epochs", type=int, default=200, help="Training epochs")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
parser.add_argument("--wd", type=float, default=1e-4, help="Weight decay")
args = parser.parse_args()

# Prepare datasets
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data")
dataset = TML1MDataset(cached_dir=path, force_reload=True)

(
    user_table,
    _,
    rating_table,
    movie_embeddings,
) = dataset.data_list
user_size = len(user_table)
emb_size = movie_embeddings.size(1)

x, ordered_rating = build_homo_data(
    relation_df=rating_table.df,
    src_col_name="UserID",
    tgt_col_name="MovieID",
    src_emb=torch.randn(user_size, emb_size),
    tgt_emb=movie_embeddings,
)

graph = build_homo_graph(
    relation_df=ordered_rating,
    x=x,
    transform=GT.GCNNorm(),
)
graph.target_table = user_table
graph.y = user_table.y.long()
graph = graph.to(device)

train_mask, val_mask, test_mask = (
    user_table.train_mask,
    user_table.val_mask,
    user_table.test_mask,
)
output_dim = user_table.num_classes


class Bridge(torch.nn.Module):
    def __init__(
        self,
        table_encoder,
        graph_encoder,
    ) -> None:
        super().__init__()
        self.table_encoder = table_encoder
        self.graph_encoder = graph_encoder

    def forward(self, target_table, x, adj):
        target_emb = self.table_encoder(target_table)
        x = torch.cat([target_emb, x[len(target_table) :, :]], dim=0)
        x = self.graph_encoder(x, adj)
        return x[: len(target_table), :]


def accuracy_score(preds, truth):
    return (preds == truth).sum(dim=0) / len(truth)


def train_epoch() -> float:
    model.train()
    optimizer.zero_grad()
    logits = model(
        target_table=graph.target_table,
        x=graph.x,
        adj=graph.adj,
    )
    loss = F.cross_entropy(logits[train_mask].squeeze(), graph.y[train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def test_epoch():
    model.eval()
    logits = model(
        target_table=graph.target_table,
        x=graph.x,
        adj=graph.adj,
    )
    preds = logits.argmax(dim=1)
    y = graph.y
    train_acc = accuracy_score(preds[train_mask], y[train_mask])
    val_acc = accuracy_score(preds[val_mask], y[val_mask])
    test_acc = accuracy_score(preds[test_mask], y[test_mask])
    return train_acc.item(), val_acc.item(), test_acc.item()


t_encoder = TableEncoder(
    hidden_dim=graph.x.size(1),
    stats_dict=user_table.stats_dict,
)
g_encoder = GraphEncoder(
    hidden_dim=graph.x.size(1),
    out_dim=output_dim,
)
model = Bridge(
    table_encoder=t_encoder,
    graph_encoder=g_encoder,
).to(device)


start_time = time.time()
best_val_acc = best_test_acc = 0
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
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

print(f"Total Time: {time.time() - start_time:.4f}s")
print(
    "Bridge result: "
    f"Best Val acc: {best_val_acc:.4f}, "
    f"Best Test acc: {best_test_acc:.4f}"
)
