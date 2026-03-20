# The GCN method from the
# "Semi-Supervised Classification with Graph Convolutional Networks" paper.
# ArXiv: https://arxiv.org/abs/1609.02907

# Datasets  CiteSeer    Cora      PubMed
# Metrics   Acc         Acc       Acc
# Rept.     0.703       0.815     0.790
# Ours      0.712       0.816     0.797
# Time      8.9s        4.0s      12.6s

import argparse
import os.path as osp
import time
import sys

import torch

sys.path.append("./")
sys.path.append("../")
from rllm.datasets import PlanetoidDataset
from rllm.transforms.graph_transforms import GCNTransform
from rllm.nn.encoder import GraphEncoder
from rllm.nn.conv.graph_conv import GCNConv

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset", type=str, default="cora", choices=["citeseer", "cora", "pubmed"]
)
parser.add_argument("--hidden_dim", type=int, default=16, help="Hidden dim")
parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
parser.add_argument("--wd", type=float, default=5e-4, help="Weight decay")
parser.add_argument("--epochs", type=int, default=200, help="Training epochs")
parser.add_argument("--dropout", type=float, default=0.5, help="Graph dropout")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

# Set random seed and device
torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
path = osp.join(osp.dirname(osp.realpath(__file__)), "..", "data")
data = PlanetoidDataset(path, args.dataset)[0]

# Transform data
transform = GCNTransform()
data = transform(data).to(device)

# Build model with GraphEncoder
model = GraphEncoder(
    in_dim=data.x.size(1),
    out_dim=data.num_classes,
    hidden_dim=args.hidden_dim,
    dropout=args.dropout,
    num_layers=2,
    graph_conv=GCNConv,
).to(device)
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=args.lr,
    weight_decay=args.wd,
)
loss_fn = torch.nn.CrossEntropyLoss()


def train() -> float:
    model.train()
    optimizer.zero_grad()
    logits = model(data.x, data.adj)
    loss = loss_fn(logits[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return float(loss.item())


@torch.no_grad()
def test():
    model.eval()
    logits = model(data.x, data.adj)
    pred = logits.argmax(dim=1)

    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        correct = float(pred[mask].eq(data.y[mask]).sum().item())
        accs.append(correct / int(mask.sum()))
    return accs


metric = "Acc"
best_val_acc = test_acc = 0.0
times = []
for epoch in range(1, args.epochs + 1):
    start = time.time()

    train_loss = train()
    train_acc, val_acc, tmp_test_acc = test()

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc

    times.append(time.time() - start)
    print(
        f"Epoch: [{epoch}/{args.epochs}] "
        f"Train Loss: {train_loss:.4f} Train {metric}: {train_acc:.4f} "
        f"Val {metric}: {val_acc:.4f}, Test {metric}: {tmp_test_acc:.4f} "
    )

print(f"Mean time per epoch: {torch.tensor(times).mean():.4f}s")
print(f"Total time: {sum(times):.4f}s")
print(f"Test {metric} at best Val: {test_acc:.4f}")
