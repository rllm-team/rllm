# The TabNet method from the
# "TabNet: Attentive Interpretable Tabular Learning" paper.
# ArXiv: https://arxiv.org/abs/1908.07442

# Datasets  Titanic     Adult
# Acc       0.843       0.853
# Time      31.1s       454.8s

import argparse
import os.path as osp
import sys
import time
from typing import Any, Dict, List

from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.append("./")
sys.path.append("../")
from rllm.types import ColType
from rllm.datasets import Adult
from rllm.nn.models import TabNet, MODEL_CONFIG

parser = argparse.ArgumentParser()
parser.add_argument("--dim", help="embedding dim", type=int, default=32)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--wd", type=float, default=5e-4)
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Prepare datasets
path = osp.join(osp.dirname(osp.realpath(__file__)), "..", "data")
dataset = Adult(cached_dir=path)[0]
# dataset = Titanic(cached_dir=path)[0]
dataset.to(device)
dataset.shuffle()

# Split dataset, here the ratio of train-val-test is 80%-10%-10%
# Split dataset, here the ratio of train-val-test is 80%-10%-10%
train_loader, val_loader, test_loader = dataset.get_dataloader(
    26048, 6513, 16281, batch_size=args.batch_size
)


# Set up model and optimizer
class TabNetModel(torch.nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        out_dim: int,
        col_stats_dict: Dict[ColType, List[Dict[str, Any]]],
    ):
        super().__init__()
        self.transform = MODEL_CONFIG[TabNet](
            out_dim=hidden_dim,
            col_stats_dict=col_stats_dict,
        )

        self.backbone = TabNet(
            out_dim=out_dim,  # dataset.num_classes,
            cat_emb_dim=hidden_dim,  # args.dim,
            num_emb_dim=hidden_dim,  # args.dim,
            col_stats_dict=col_stats_dict,
        )

    def forward(self, x):
        x = self.transform(x)
        out = self.backbone(x.reshape(x.size(0), -1))
        return out


model = TabNetModel(
    out_dim=dataset.num_classes,
    hidden_dim=args.dim,
    col_stats_dict=dataset.metadata,
).to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=args.lr,
    weight_decay=args.wd,
)


def train(epoch: int, lambda_sparse: float = 1e-4) -> float:
    model.train()
    loss_accum = total_count = 0
    for batch in tqdm(train_loader, desc=f"Epoch: {epoch}"):
        x, y = batch
        pred, M_loss = model.forward(x)
        loss = F.cross_entropy(pred, y.long())
        loss = loss - lambda_sparse * M_loss
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
        pred, _ = model.forward(feat_dict)
        _, predicted = torch.max(pred, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()
    accuracy = correct / total
    return accuracy


metric = "Acc"
best_val_metric = 0
best_test_metric = 0
times = []
st = time.time()
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
    optimizer.step()
et = time.time()
print(f"Mean time per epoch: {torch.tensor(times).mean():.4f}s")
print(f"Total time: {et-st}s")
print(
    f"Best Val {metric}: {best_val_metric:.4f}, "
    f"Best Test {metric}: {best_test_metric:.4f}"
)
