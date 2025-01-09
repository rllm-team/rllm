# The GCN method from the
# "Semi-Supervised Classification with Graph Convolutional Networks" paper.
# ArXiv: https://arxiv.org/abs/1609.02907

# Datasets  CiteSeer    Cora      PubMed
# Acc       0.712       0.816     0.787
# Time      8.9s        4.0s      12.6s

import argparse
import os.path as osp
import time
import sys

import torch
import torch.nn.functional as F

sys.path.append("./")
sys.path.append("../")
from rllm.datasets import PlanetoidDataset
from rllm.transforms.graph_transforms import GCNTransform
from rllm.nn.conv.graph_conv import GCNConv

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset", type=str, default="cora", choices=["citeseer", "cora", "pubmed"]
)
parser.add_argument("--hidden_dim", type=int, default=16, help="Hidden dim")
parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
parser.add_argument("--wd", type=float, default=5e-4, help="Weight decay")
parser.add_argument("--epochs", type=int, default=200, help="Training epochs")
parser.add_argument("--dropout", type=float, default=0.5, help="Graph Dropout")
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


# Define model
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


# Set up model, optimizer and loss function
model = GCN(
    in_dim=data.x.shape[1],
    hidden_dim=args.hidden_dim,
    out_dim=data.num_classes,
    dropout=args.dropout,
).to(device)
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=args.lr,
    weight_decay=args.wd,
)
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


metric = "Acc"
best_val_acc = best_test_acc = 0
times = []
for epoch in range(1, args.epochs + 1):
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
