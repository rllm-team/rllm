# The BRIDGE method from the "rLLM: Relational Table Learning with LLMs" paper.
# ArXiv: https://arxiv.org/abs/2407.20157

# Datasets  TACM12K
# Acc       0.309

import time
import argparse
import os.path as osp
import sys

import torch
import torch.nn.functional as F

sys.path.append("./")
sys.path.append("../")
from rllm.datasets import TACM12KDataset
import rllm.transforms.graph_transforms as GT
from rllm.nn.conv.graph_conv import GCNConv
from rllm.nn.conv.table_conv import TabTransformerConv
from utils import build_homo_adj, TableEncoder, GraphEncoder


parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
parser.add_argument("--wd", type=float, default=5e-4, help="Weight decay")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

# Prepare datasets
torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data")
dataset = TACM12KDataset(cached_dir=path, force_reload=True)

(
    papers_table,
    authors_table,
    citations_table,
    _,
    paper_embeddings,
    _,
) = dataset.data_list

# Making graph
paper_embeddings = paper_embeddings.to(device)
adj = build_homo_adj(
    relation_df=citations_table.df,
    n_all=len(papers_table),
).to(device)
target_table = papers_table.to(device)
y = papers_table.y.long().to(device)


train_mask, val_mask, test_mask = (
    papers_table.train_mask,
    papers_table.val_mask,
    papers_table.test_mask,
)
out_dim = papers_table.num_classes


class Bridge(torch.nn.Module):
    def __init__(
        self,
        table_encoder,
        graph_encoder,
    ) -> None:
        super().__init__()
        self.table_encoder = table_encoder
        self.graph_encoder = graph_encoder

    def forward(self, table, non_table, adj):
        t_embedds = self.table_encoder(table)
        node_feats = torch.cat([t_embedds, non_table], dim=0)
        node_feats = self.graph_encoder(node_feats, adj)
        return node_feats[: len(table), :]


def accuracy_score(preds, truth):
    return (preds == truth).sum(dim=0) / len(truth)


def train_epoch() -> float:
    model.train()
    optimizer.zero_grad()
    logits = model(
        table=target_table,
        non_table=paper_embeddings[len(target_table) :, :],
        adj=adj,
    )
    loss = F.cross_entropy(logits[train_mask].squeeze(), y[train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def test_epoch():
    model.eval()
    logits = model(
        table=target_table,
        non_table=paper_embeddings[len(target_table) :, :],
        adj=adj,
    )
    preds = logits.argmax(dim=1)
    train_acc = accuracy_score(preds[train_mask], y[train_mask])
    val_acc = accuracy_score(preds[val_mask], y[val_mask])
    test_acc = accuracy_score(preds[test_mask], y[test_mask])
    return train_acc.item(), val_acc.item(), test_acc.item()


t_encoder = TableEncoder(
    out_dim=paper_embeddings.size(1),
    stats_dict=papers_table.stats_dict,
    table_conv=TabTransformerConv,
)
g_encoder = GraphEncoder(
    in_dim=paper_embeddings.size(1),
    out_dim=out_dim,
    graph_transform=GT.GCNNorm(),
    graph_conv=GCNConv,
)
model = Bridge(
    table_encoder=t_encoder,
    graph_encoder=g_encoder,
).to(device)

start_time = time.time()
best_val_acc = best_test_acc = 0
optimizer = torch.optim.Adam(
    [
        dict(params=model.table_encoder.parameters(), lr=0.001),
        dict(params=model.graph_encoder.parameters(), lr=0.01, weight_decay=1e-4),
    ]
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

print(f"Total Time: {time.time() - start_time:.4f}s")
print(
    "Bridge result: "
    f"Best Val acc: {best_val_acc:.4f}, "
    f"Best Test acc: {best_test_acc:.4f}"
)
