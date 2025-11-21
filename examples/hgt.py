# The HGT method from the
# "Heterogeneous Graph Transformer" paper.
# ArXiv: https://arxiv.org/abs/2003.01332

# Datasets  IMDB        DBLP
# Metrics   Macro-F1    Macro-F1
# Rept.     -           -
# Ours      0.579       0.786
# Time      8.145s      19.758s

import argparse
import sys
import time
from typing import Dict, List, Union
import os.path as osp

import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

sys.path.append("./")
sys.path.append("../")
from rllm.datasets import IMDB, DBLP
from rllm.nn.conv.graph_conv import HGTConv

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="imdb", choices=["imdb", "dblp"])
parser.add_argument("--lr", type=float, default=5e-3, help="Learning rate")
parser.add_argument("--wd", type=float, default=1e-3, help="Weight decay")
parser.add_argument("--dropout", type=float, default=0.6, help="Graph Dropout")
parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
args = parser.parse_args()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
path = osp.join(osp.dirname(osp.realpath(__file__)), "..", "data")
if args.dataset.lower() == "dblp":
    data = DBLP(cached_dir=path)[0]
    data["conference"].x = torch.full(
        (data["conference"].num_nodes, 1), 1, dtype=torch.float
    )
    target_node_type = "author"
else:
    data = IMDB(cached_dir=path)[0]
    target_node_type = "movie"
data.to(device)


# Define model
class HGT(torch.nn.Module):
    def __init__(
        self,
        in_dim: Union[int, Dict[str, int]],
        out_dim: int,
        hidden_dim: int = 128,
        num_heads: int = 8,
        dropout: int = 0.6,
        metadata: Dict[str, List[str]] = None,
    ):
        super().__init__()
        self.hgt_conv = HGTConv(
            in_dim=in_dim,
            out_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            metadata=metadata,
            use_pre_encoder=True,
        )
        self.lin = torch.nn.Linear(hidden_dim, out_dim)

    def forward(self, x_dict, adj_dict):
        out = self.hgt_conv(x_dict, adj_dict)
        out = self.lin(out[target_node_type])
        return out


# Set up model and optimizer
in_dim = {node_type: data[node_type].x.shape[1] for node_type in data.node_types}
out_dim = torch.unique(data[target_node_type].y).numel()
model = HGT(
    in_dim=in_dim,
    out_dim=out_dim,
    dropout=args.dropout,
    metadata=data.metadata(),
).to(device)
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=args.lr,
    weight_decay=args.wd,
)


def train() -> float:
    model.train()
    optimizer.zero_grad()
    out = model(data.x_dict(), data.adj_dict())
    mask = data.train_mask
    loss = F.cross_entropy(out[mask], data[target_node_type].y[mask])
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test() -> List[float]:
    model.eval()
    pred = model(data.x_dict(), data.adj_dict()).argmax(dim=-1)

    f1s = []
    for split in ["train_mask", "val_mask", "test_mask"]:
        mask = getattr(data, split)
        y_true = data[target_node_type].y[mask].cpu().numpy()
        y_pred = pred[mask].cpu().numpy()
        f1 = f1_score(y_true, y_pred, average="macro")
        f1s.append(float(f1))
    return f1s


metric = "Macro-F1"
best_val_acc = test_acc = 0
times = []
start_patience = patience = args.patience
for epoch in range(1, args.epochs + 1):
    start = time.time()

    train_loss = train()
    train_acc, val_acc, tmp_test_acc = test()

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc
        patience = start_patience
    else:
        patience -= 1

    times.append(time.time() - start)
    print(
        f"Epoch: [{epoch}/{args.epochs}] "
        f"Train Loss: {train_loss:.4f} Train {metric}: {train_acc:.4f} "
        f"Val {metric}: {val_acc:.4f}, Test {metric}: {tmp_test_acc:.4f} "
    )

    if patience <= 0:
        print(
            "Stopping training as validation accuracy did not improve "
            f"for {start_patience} epochs"
        )
        break

print(f"Mean time per epoch: {torch.tensor(times).mean():.4f}s")
print(f"Total time: {sum(times):.4f}s")
print(f"Test {metric} at best Val: {test_acc:.4f}")
