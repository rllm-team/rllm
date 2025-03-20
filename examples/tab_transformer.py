# The TabTransformer method from the
# "TabTransformer: Tabular Data Modeling Using Contextual Embeddings" paper.
# ArXiv: https://arxiv.org/abs/2012.06678

# Datasets  Titanic    Adult
# Acc       0.842      0.842
# Time      5.26s      152.9s

import argparse
import sys
import time
from typing import Any, Dict, List
import os.path as osp

from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.append("./")
sys.path.append("../")
from rllm.types import ColType
from rllm.datasets import Titanic
from rllm.transforms.table_transforms import TabTransformerTransform
from rllm.nn.conv.table_conv import TabTransformerConv

parser = argparse.ArgumentParser()
parser.add_argument("--emb_dim", help="embedding dim", type=int, default=32)
parser.add_argument("--num_layers", type=int, default=2)
parser.add_argument("--num_heads", type=int, default=8)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--wd", type=float, default=5e-4)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

# Set random seed and device
torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
path = osp.join(osp.dirname(osp.realpath(__file__)), "..", "data")
data = Titanic(cached_dir=path)[0]

# Transform data
transform = TabTransformerTransform(out_dim=args.emb_dim)
data = transform(data).to(device)
data.shuffle()

# Split dataset, here the ratio of train-val-test is 80%-10%-10%
train_loader, val_loader, test_loader = data.get_dataloader(
    train_split=0.8, val_split=0.1, test_split=0.1, batch_size=args.batch_size
)


# Define model
class TabTransformer(torch.nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        out_dim: int,
        num_layers: int,
        num_heads: int,
        metadata: Dict[ColType, List[Dict[str, Any]]],
    ):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(
            TabTransformerConv(
                conv_dim=hidden_dim,
                num_heads=num_heads,
                use_pre_encoder=True,
                metadata=metadata,
            )
        )
        for _ in range(num_layers - 1):
            self.convs.append(
                TabTransformerConv(conv_dim=hidden_dim, num_heads=num_heads)
            )

        self.fc = torch.nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        x = torch.cat(list(x.values()), dim=1)
        out = self.fc(x.mean(dim=1))
        return out


# Set up model and optimizer
model = TabTransformer(
    hidden_dim=args.emb_dim,
    out_dim=data.num_classes,
    num_layers=args.num_layers,
    num_heads=args.num_heads,
    metadata=data.metadata,
).to(device)
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=args.lr,
    weight_decay=args.wd,
)


def train(epoch: int) -> float:
    model.train()
    loss_accum = total_count = 0.0
    for batch in tqdm(train_loader, desc=f"Epoch: {epoch}"):
        x, y = batch
        pred = model.forward(x)
        loss = F.cross_entropy(pred, y.long())
        optimizer.zero_grad()
        loss.backward()
        loss_accum += float(loss) * y.size(0)
        total_count += y.size(0)
        optimizer.step()
    return loss_accum / total_count


@torch.no_grad()
def test(loader: DataLoader) -> float:
    model.eval()
    correct = total = 0
    for batch in loader:
        feat_dict, y = batch
        pred = model.forward(feat_dict)
        _, predicted = torch.max(pred, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()
    accuracy = correct / total
    return accuracy


metric = "Acc"
best_val_metric = best_test_metric = 0
times = []
for epoch in range(1, args.epochs + 1):
    start = time.time()

    train_loss = train(epoch)
    train_metric = test(train_loader)
    val_metric = test(val_loader)
    test_metric = test(test_loader)

    if val_metric > best_val_metric:
        best_val_metric = val_metric
        best_test_metric = test_metric

    times.append(time.time() - start)
    print(
        f"Train Loss: {train_loss:.4f}, Train {metric}: {train_metric:.4f}, "
        f"Val {metric}: {val_metric:.4f}, Test {metric}: {test_metric:.4f}"
    )

print(f"Mean time per epoch: {torch.tensor(times).mean():.4f}s")
print(f"Total time: {sum(times):.4f}s")
print(
    f"Best Val {metric}: {best_val_metric:.4f}, "
    f"Best Test {metric}: {best_test_metric:.4f}"
)
