# The GCN method from the
# "Semi-Supervised Classification with Graph Convolutional Networks" paper.
# ArXiv: https://arxiv.org/abs/1609.02907

# Datasets  CiteSeer    Cora      PubMed
# Acc       0.712       0.816     0.787
# Time      8.9s        4.0s      12.6s

import argparse
from ast import arg
import os.path as osp
import time

import torch
import torch.nn.functional as F

import sys

sys.path.append("../")
import rllm.transforms.graph_transforms as T
from rllm.datasets.planetoid import PlanetoidDataset
from rllm.nn.conv.graph_conv import GCNConv

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset", type=str, default="cora", choices=["citeseer", "cora", "pubmed"]
)
parser.add_argument("--hidden_dim", type=int, default=16, help="Hidden channel")
parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
parser.add_argument("--wd", type=float, default=5e-4, help="Weight decay")
parser.add_argument("--epochs", type=int, default=200, help="Training epochs")
parser.add_argument("--dropout", type=float, default=0.5, help="Graph Dropout")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

torch.manual_seed(args.seed)
transform = T.Compose([T.NormalizeFeatures("l2"), T.GCNNorm()])

path = osp.join(osp.dirname(osp.realpath(__file__)), "..", "data")
dataset = PlanetoidDataset(path, args.dataset, transform=transform)
data = dataset[0]


class GCN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout):
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


model = GCN(
    in_dim=data.x.shape[1],
    hidden_dim=args.hidden_dim,
    out_dim=data.num_classes,
    dropout=args.dropout,
)

optimizer = torch.optim.Adam(
    model.parameters(), lr=args.lr, weight_decay=args.wd
)  # Only perform weight-decay on first convolution.
loss_fn = torch.nn.CrossEntropyLoss()


def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.adj)
    loss = loss_fn(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def test():
    model.eval()
    out = model(data.x, data.adj)
    pred = out.argmax(dim=1)

    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        correct = float(pred[mask].eq(data.y[mask]).sum().item())
        accs.append(correct / int(mask.sum()))
    return accs


best_val_acc = best_test_acc = 0
times = []
st = time.time()
for epoch in range(1, args.epochs + 1):
    start = time.time()
    train_loss = train()
    train_acc, val_acc, test_acc = test()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_test_acc = test_acc
    times.append(time.time() - start)
    print(
        f"Epoch: [{epoch}/{args.epochs}]"
        f"Loss: {train_loss:.4f} train_acc: {train_acc:.4f} "
        f"val_acc: {val_acc:.4f} test_acc: {test_acc:.4f} "
    )
et = time.time()
print(f"Mean time per epoch: {torch.tensor(times).mean():.4f}s")
print(f"Total time: {et-st}s")
print(f"Best test acc: {best_test_acc:.4f}")
