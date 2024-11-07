# The BRIDGE method from the "rLLM: Relational Table Learning with LLMs" paper.
# ArXiv: https://arxiv.org/abs/2407.20157

# Datasets  TLF2K
# Acc       0.494

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
from rllm.datasets import TLF2KDataset
from utils import get_homo_data, build_homo_graph, GraphEncoder, TableEncoder


parser = argparse.ArgumentParser()
parser.add_argument("--gcn_dropout", type=float, default=0.5, help="Dropout for GCN")
parser.add_argument("--epochs", type=int, default=200, help="Training epochs")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
parser.add_argument("--wd", type=float, default=1e-4, help="Weight decay")
args = parser.parse_args()

# Prepare datasets
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data")
dataset = TLF2KDataset(cached_dir=path, force_reload=True)

artist_table, ua_table, _ = dataset.data_list
artist_size = len(artist_table)
user_size = ua_table.df["userID"].max()
emb_size = 384

x, ordered_ua = get_homo_data(
    relation_df=ua_table.df,
    src_col_name="artistID",
    tgt_col_name="userID",
    src_emb=torch.randn(artist_size, emb_size),
    tgt_emb=torch.randn(user_size, emb_size),
)

graph = build_homo_graph(
    relation_df=ordered_ua,
    x=x,
    transform=GT.GCNNorm(),
)
graph.y = artist_table.y.long()
graph.target_table = artist_table
graph = graph.to(device)

train_mask, val_mask, test_mask = (
    artist_table.train_mask,
    artist_table.val_mask,
    artist_table.test_mask,
)
output_dim = artist_table.num_classes


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
    stats_dict=artist_table.stats_dict,
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
