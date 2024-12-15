# The HGT method from the
# "Heterogeneous Graph Transformer" paper.
# ArXiv: https://arxiv.org/abs/2003.01332

# Datasets  IMDB
# Acc       0.583

import sys
import time
from typing import List
import os.path as osp

import torch
import torch.nn.functional as F

sys.path.append("./")
sys.path.append("../")
from rllm.datasets import IMDB
from rllm.data import HeteroGraphData
from rllm.nn.conv.graph_conv import HGTConv

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
path = osp.join(osp.dirname(osp.realpath(__file__)), "..", "data")
dataset = IMDB(path)
data = dataset[0]
data.to(device)


class HGT(torch.nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        out_dim: int,
        data: HeteroGraphData,
        heads=8,
    ):
        super().__init__()
        self.lin_dict = torch.nn.ModuleDict()
        for node_type in data.node_types:
            self.lin_dict[node_type] = torch.nn.Linear(
                in_features=data.x_dict()[node_type].shape[1],
                out_features=hidden_dim,
            )
        self.hgt_conv = HGTConv(
            hidden_dim,
            hidden_dim,
            heads=heads,
            dropout_rate=0.6,
            metadata=data.metadata(),
        )
        self.lin = torch.nn.Linear(hidden_dim, out_dim)

    def forward(self, x_dict, adj_dict):
        out = {}
        for node_type, x in x_dict.items():
            out[node_type] = self.lin_dict[node_type](x)
        out = self.hgt_conv(out, adj_dict)
        out = self.lin(out["movie"])
        return out


model = HGT(
    data=data,
    hidden_dim=128,
    out_dim=3,
).to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.005,
    weight_decay=0.001,
)


def train() -> float:
    model.train()
    optimizer.zero_grad()
    out = model(data.x_dict(), data.adj_dict())
    mask = data.train_mask
    loss = F.cross_entropy(out[mask], data["movie"].y[mask])
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test() -> List[float]:
    model.eval()
    pred = model(data.x_dict(), data.adj_dict()).argmax(dim=-1)

    accs = []
    for split in ["train_mask", "val_mask", "test_mask"]:
        # mask = data[split]
        mask = getattr(data, split)
        acc = (pred[mask] == data["movie"].y[mask]).sum() / mask.sum()
        accs.append(float(acc))
    return accs


metric = "Acc"
best_val_acc = best_test_acc = 0
times = []
for epoch in range(1, 201):
    start = time.time()
    train_loss = train()
    train_acc, val_acc, test_acc = test()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_test_acc = test_acc
    times.append(time.time() - start)
    print(
        f"Epoch: [{epoch}/{args.epochs}] "
        f"Train Loss: {train_loss:.4f} Train {metric}: {train_acc:.4f} "
        f"Val {metric}: {val_acc:.4f}, Test {metric}: {test_acc:.4f} "
    )
print(f"Mean time per epoch: {torch.tensor(times).mean():.4f}s")
print(f"Total time: {sum(times):.4f}s")
print(f"Best test acc: {best_test_acc:.4f}")
