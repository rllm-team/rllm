# The BRIDGE method from the "rLLM: Relational Table Learning with LLMs" paper.
# ArXiv: https://arxiv.org/abs/2407.20157

# Datasets  TLF2K
# Acc       0.494

import time
import argparse
import os.path as osp
import sys

sys.path.append("../")
sys.path.append("../../")

import torch
import torch.nn.functional as F

import rllm.transforms.graph_transforms as T
from rllm.transforms.table_transforms import FTTransformerTransform
from rllm.nn.conv.table_conv import TabTransformerConv
from rllm.nn.conv.graph_conv import GCNConv
from rllm.datasets import TLF2KDataset
from utils import build_homo_graph


parser = argparse.ArgumentParser()
parser.add_argument(
    "--tab_dim", type=int, default=64, help="Tab Transformer categorical embedding dim"
)
parser.add_argument("--gcn_dropout", type=float, default=0.5, help="Dropout for GCN")
parser.add_argument("--epochs", type=int, default=200, help="Training epochs")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
parser.add_argument("--wd", type=float, default=1e-4, help="Weight decay")
args = parser.parse_args()

# Prepare datasets
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data")

dataset = TLF2KDataset(cached_dir=path, force_reload=True)
artist_table, ua_table, uu_table = dataset.data_list

# We assume it a homogeneous graph,
# so we need to reorder the user and artist id.
ordered_ua = ua_table.df.assign(
    artistID=ua_table.df["artistID"] - 1,
    userID=ua_table.df["userID"] + len(artist_table) - 1,
)

# Making graph
emb_size = 384  # Since user doesn't have an embedding, randomly select a dim.
len_artist = len(artist_table)
len_user = ua_table.df["userID"].max()
# Randomly initialize the embedding, artist embedding will be further trained
x = torch.randn(len_artist + len_user, emb_size)
graph = build_homo_graph(
    df=ordered_ua,
    n_src=len_artist,
    n_tgt=len_user,
    x=x,
    y=artist_table.y.long(),
    transform=T.GCNNorm(),
)
graph.artist_table = artist_table
graph = graph.to(device)
train_mask, val_mask, test_mask = (
    graph.artist_table.train_mask,
    graph.artist_table.val_mask,
    graph.artist_table.test_mask,
)
output_dim = graph.artist_table.num_classes


class GraphEncoder(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout) -> None:
        super().__init__()
        self.dropout = dropout
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)

    def forward(self, x, adj):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv1(x, adj))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, adj)
        return x


class TableEncoder(torch.nn.Module):
    def __init__(
        self,
        hidden_dim,
        stats_dict,
    ) -> None:
        super().__init__()
        self.table_transform = FTTransformerTransform(
            out_dim=hidden_dim,
            col_stats_dict=stats_dict,
        )
        self.conv = TabTransformerConv(
            dim=hidden_dim,
        )

    def forward(self, table):
        feat_dict = table.get_feat_dict()  # A dict contains feature tensor.
        x, _ = self.table_transform(feat_dict)
        x = self.conv(x)
        x = x.mean(dim=1)
        return x


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
        graph.artist_table, graph.x, graph.adj, len_artist, len_artist + len_user
    )
    loss = F.cross_entropy(logits[train_mask].squeeze(), graph.y[train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def test_epoch():
    model.eval()
    logits = model(
        graph.artist_table, graph.x, graph.adj, len_artist, len_artist + len_user
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
)
g_encoder = GraphEncoder(
    in_dim=graph.x.size(1),
    hidden_dim=128,
    out_dim=output_dim,
    dropout=args.gcn_dropout,
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
