# The TabTransformer method from the
# "TabTransformer: Tabular Data Modeling Using Contextual Embeddings" paper.
# ArXiv: https://arxiv.org/abs/2012.06678

# Datasets  Titanic    Adult
# AUC       0.809      0.839
# Time      11.3s      391.1s

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
from rllm.datasets import Titanic
from rllm.nn.models import TNNConfig
from rllm.nn.conv.table_conv import TabTransformerConv

parser = argparse.ArgumentParser()
parser.add_argument("--dim", help="transform dim", type=int, default=32)
parser.add_argument("--num_layers", type=int, default=2)
parser.add_argument("--num_heads", type=int, default=8)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--wd", type=float, default=5e-4)
args = parser.parse_args()

torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Prepare datasets
transform = TNNConfig.get_transform("TabTransformer")(args.dim)
path = osp.join(osp.dirname(osp.realpath(__file__)), "..", "data")
dataset = Titanic(cached_dir=path, transform=transform)[0]
# transform
# 如果使用FTTransformEncoder
# dataset = Titanic(cached_dir=path)[0]
dataset.to(device)

# Split dataset, here the ratio of train-val-test is 80%-10%-10%
train_loader, val_loader, test_loader = dataset.get_dataloader(
    0.8, 0.1, 0.1, batch_size=args.batch_size
)


class TabTransformer(torch.nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        out_dim: int,
        num_layers: int,  # 这个参数没用啊
        heads: int,
        metadata: Dict[ColType, List[Dict[str, Any]]],
    ):
        super().__init__()
        self.pre_encoder = TNNConfig.get_pre_encoder("TabTransformer")(
            out_dim=hidden_dim,
            metadata=metadata,
        )
        # 使用FTTransformEncoder Best AUC 0.7753; 原来的结构Best AUC 0.8864
        # self.pre_encoder = FTTransformerEncoder(
        #     out_dim=hidden_dim,
        #     metadata=metadata,
        # )

        self.convs = torch.nn.ModuleList(
            [
                TabTransformerConv(
                    dim=hidden_dim,
                    heads=heads,
                    pre_encoder=self.pre_encoder,
                ),
                TabTransformerConv(
                    dim=hidden_dim,
                    heads=heads,
                ),
            ]
        )
        self.fc = torch.nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        out = self.fc(x.mean(dim=1))
        return out


model = TabTransformer(
    hidden_dim=args.dim,
    out_dim=dataset.num_classes,
    num_layers=args.num_layers,
    heads=args.num_heads,
    metadata=dataset.metadata,
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
        loss_accum += float(loss) * y.size(0)  # daigai
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
