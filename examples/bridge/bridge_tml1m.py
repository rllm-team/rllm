# The BRIDGE method from the "rLLM: Relational Table Learning with LLMs" paper.
# ArXiv: https://arxiv.org/abs/2407.20157

# Datasets  TML1M
# Acc       0.428

import time
import argparse
import sys
import os.path as osp

import torch
import torch.nn.functional as F

sys.path.append("./")
sys.path.append("../")
from rllm.datasets import TML1MDataset
from rllm.transforms.graph_transforms import GCNTransform
from rllm.transforms.table_transforms import TabTransformerTransform
from rllm.nn.conv.graph_conv import GCNConv
from rllm.nn.conv.table_conv import TabTransformerConv
from utils import build_homo_graph, reorder_ids, TableEncoder, GraphEncoder


parser = argparse.ArgumentParser()
parser.add_argument("--gcn_dropout", type=float, default=0.5, help="Dropout for GCN")
parser.add_argument("--epochs", type=int, default=200, help="Training epochs")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
parser.add_argument("--wd", type=float, default=1e-4, help="Weight decay")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data")
dataset = TML1MDataset(cached_dir=path, force_reload=True)

(
    user_table,
    _,
    rating_table,
    movie_embeddings,
) = dataset.data_list
emb_size = movie_embeddings.size(1)
user_size = len(user_table)

ordered_rating = reorder_ids(
    relation_df=rating_table.df,
    src_col_name="UserID",
    tgt_col_name="MovieID",
    n_src=user_size,
)
target_table = user_table.to(device)
y = user_table.y.long().to(device)
movie_embeddings = movie_embeddings.to(device)

graph = build_homo_graph(
    relation_df=ordered_rating,
    n_all=user_size + movie_embeddings.size(0),
).to(device)

# Transform data
table_transform = TabTransformerTransform(
    out_dim=emb_size, metadata=target_table.metadata
)
target_table = table_transform(target_table)
graph_transform = GCNTransform()
adj = graph_transform(graph).adj

train_mask, val_mask, test_mask = (
    user_table.train_mask,
    user_table.val_mask,
    user_table.test_mask,
)


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


t_encoder = TableEncoder(
    in_dim=emb_size,
    out_dim=emb_size,
    table_conv=TabTransformerConv,
    metadata=target_table.metadata,
)
g_encoder = GraphEncoder(
    in_dim=emb_size,
    out_dim=user_table.num_classes,
    graph_conv=GCNConv,
)
model = Bridge(
    table_encoder=t_encoder,
    graph_encoder=g_encoder,
).to(device)
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=args.lr,
    weight_decay=args.wd,
)


def accuracy_score(preds, truth):
    return (preds == truth).sum(dim=0) / len(truth)


def train_epoch() -> float:
    model.train()
    optimizer.zero_grad()
    logits = model(
        table=user_table,
        non_table=movie_embeddings,
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
        table=user_table,
        non_table=movie_embeddings,
        adj=adj,
    )
    preds = logits.argmax(dim=1)
    train_acc = accuracy_score(preds[train_mask], y[train_mask])
    val_acc = accuracy_score(preds[val_mask], y[val_mask])
    test_acc = accuracy_score(preds[test_mask], y[test_mask])
    return train_acc.item(), val_acc.item(), test_acc.item()


start_time = time.time()
best_val_acc = best_test_acc = 0
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
