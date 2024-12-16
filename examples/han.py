# The HAN method from the
# "Heterogeneous Graph Attention Network" paper.
# ArXiv: https://arxiv.org/abs/1903.07293

# Datasets  IMDB
# Acc       0.571

import sys
import time
from typing import Dict, List, Union
import os.path as osp

import torch
import torch.nn.functional as F

sys.path.append("./")
sys.path.append("../")
from rllm.datasets import IMDB
from rllm.nn.conv.graph_conv import HANConv

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
path = osp.join(osp.dirname(osp.realpath(__file__)), "..", "data")
dataset = IMDB(path)
data = dataset[0]
data.to(device)


class HAN(torch.nn.Module):
    def __init__(
        self,
        in_dim: Union[int, Dict[str, int]],
        out_dim: int,
        hidden_dim=128,
        heads=8,
        metadata: Dict[str, List[str]] = None,
    ):
        super().__init__()
        self.han_conv = HANConv(
            in_dim=in_dim,
            out_dim=hidden_dim,
            heads=heads,
            dropout=0.6,
            metadata=metadata,
            use_pre_encoder=True,
        )
        self.lin = torch.nn.Linear(hidden_dim, out_dim)

    def forward(self, x_dict, adj_dict):
        out = self.han_conv(x_dict, adj_dict)
        out = self.lin(out["movie"])
        return out


in_dim = {node_type: data[node_type].x.shape[1] for node_type in data.node_types}

# Set up model and optimizer
model = HAN(
    in_dim=in_dim,
    out_dim=3,
    metadata=data.metadata(),
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
        mask = getattr(data, split)
        acc = (pred[mask] == data["movie"].y[mask]).sum() / mask.sum()
        accs.append(float(acc))
    return accs


metric = "Acc"
best_val_acc = best_test_acc = 0
times = []
for epoch in range(1, 51):
    start = time.time()
    train_loss = train()
    train_acc, val_acc, test_acc = test()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_test_acc = test_acc
    times.append(time.time() - start)
    print(
        f"Epoch: [{epoch}/{200}] "
        f"Train Loss: {train_loss:.4f} Train {metric}: {train_acc:.4f} "
        f"Val {metric}: {val_acc:.4f}, Test {metric}: {test_acc:.4f} "
    )
print(f"Mean time per epoch: {torch.tensor(times).mean():.4f}s")
print(f"Total time: {sum(times):.4f}s")
print(f"Best test acc: {best_test_acc:.4f}")
