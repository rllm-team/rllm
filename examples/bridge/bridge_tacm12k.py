# The BRIDGE method from the "rLLM: Relational Table Learning with LLMs" paper.
# ArXiv: https://arxiv.org/abs/2407.20157

# Datasets  TACM12K
# Acc       0.324

import time
import argparse
import os.path as osp
import sys


sys.path.append("./")
sys.path.append("../")
sys.path.append("../../")

import torch
import torch.nn.functional as F

from rllm.transforms.table_transforms import FTTransformerTransform
import rllm.transforms.graph_transforms as T
from rllm.nn.conv.table_conv import TabTransformerConv
from rllm.nn.conv.graph_conv import GCNConv
from rllm.datasets import TACM12KDataset
from utils import build_homo_graph, GraphEncoder, TableEncoder


parser = argparse.ArgumentParser()
parser.add_argument(
    "--tab_dim", type=int, default=256, help="TabTransformer categorical embedding dim"
)
parser.add_argument("--gcn_dropout", type=float, default=0.5, help="Dropout for GCN")
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

paper_table, author_table, citation, _, paper_emb, _ = dataset.data_list
# get the homogeneous data converted from the original data
# x, relation_df = dataset.homo_data()

# Making graph
graph = build_homo_graph(
    relation_df=citation.df,
    x=paper_emb,
    transform=T.GCNNorm(),
)
graph.target_table = paper_table
graph.y = paper_table.y.long()
graph = graph.to(device)

train_mask, val_mask, test_mask = (
    paper_table.train_mask,
    paper_table.val_mask,
    paper_table.test_mask,
)
output_dim = paper_table.num_classes


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
    stats_dict=paper_table.stats_dict,
    table_transorm=FTTransformerTransform,
    table_conv=TabTransformerConv,
    conv_params={
        "attn_dropout": 0.3,
        "ff_dropout": 0.3,
    },
)
g_encoder = GraphEncoder(
    in_dim=graph.x.size(1),
    hidden_dim=128,
    out_dim=output_dim,
    dropout=args.gcn_dropout,
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
