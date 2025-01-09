# The ExcelFormer method from the
# "ExcelFormer: A neural network surpassing GBDTs on tabular data" paper.
# ArXiv: https://arxiv.org/abs/2301.02819

# Datasets  Titanic    Adult
# AUC       0.920      0.913
# Time      7.3s       231.1s

import argparse
import sys
import time
from typing import Any, Dict, List
import os.path as osp

from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import torch
from torch import Tensor
from torch.utils.data import DataLoader
import torch.nn.functional as F

sys.path.append("./")
sys.path.append("../")
from rllm.types import ColType
from rllm.datasets.titanic import Titanic
from rllm.transforms.table_transforms import DefaultTableTransform
from rllm.nn.conv.table_conv import ExcelFormerConv

parser = argparse.ArgumentParser()
parser.add_argument("--emb_dim", help="embedding dim.", type=int, default=32)
parser.add_argument("--num_layers", type=int, default=3)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--lr", type=float, default=1e-3)
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
transform = DefaultTableTransform(out_dim=args.emb_dim)
data = transform(data).to(device)
data.shuffle()

# Split dataset, here the ratio of train-val-test is 80%-10%-10%
train_loader, val_loader, test_loader = data.get_dataloader(
    train_split=0.8, val_split=0.1, test_split=0.1, batch_size=args.batch_size
)


# Define model
class ExcelFormer(torch.nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        out_dim: int,
        num_layers: int,
        metadata: Dict[ColType, List[Dict[str, Any]]],
    ):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(
            ExcelFormerConv(
                conv_dim=hidden_dim,
                use_pre_encoder=True,
                metadata=metadata,
            )
        )
        for _ in range(num_layers - 1):
            self.convs.append(ExcelFormerConv(conv_dim=hidden_dim))

        self.fc = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x) -> Tensor:
        for conv in self.convs:
            x = conv(x)
        out = self.fc(x.mean(dim=1))
        return out


# Set up model and optimizer
model = ExcelFormer(
    hidden_dim=args.emb_dim,
    out_dim=data.num_classes,
    num_layers=args.num_layers,
    metadata=data.metadata,
).to(device)
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=args.lr,
    weight_decay=args.wd,
)


def train(epoch: int) -> float:
    model.train()
    loss_accum = total_count = 0
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
    all_preds = []
    all_labels = []
    for batch in loader:
        x, y = batch
        pred = model.forward(x)
        all_labels.append(y.cpu())
        all_preds.append(pred[:, 1].detach().cpu())
    all_labels = torch.cat(all_labels).numpy()
    all_preds = torch.cat(all_preds).numpy()

    # Compute the overall AUC
    overall_auc = roc_auc_score(all_labels, all_preds)
    return overall_auc


metric = "AUC"
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
