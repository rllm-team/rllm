# The Transtab method from the
# "TransTab: Learning Transferable Tabular Transformers Across Tables" paper.
# ArXiv: https://arxiv.org/abs/2205.09328

# Datasets  Titanic    Adult
# AUC       0.843      0.908
# Time      35.0s      154.6s

import argparse
import sys
import time
from typing import List
import os.path as osp

from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.append("./")
sys.path.append("../")
from rllm.types import ColType
from rllm.datasets import Titanic
from rllm.nn.models import TransTabClassifier

parser = argparse.ArgumentParser()
parser.add_argument("--hidden_dim", type=int, default=128, help="Transformer hidden dim")
parser.add_argument("--num_layers", type=int, default=2, help="Number of transformer layers")
parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
parser.add_argument("--wd", type=float, default=1e-4, help="Weight decay")
parser.add_argument("--seed", type=int, default=123, help="Random seed")
args = parser.parse_args()

# Set random seed and device
torch.manual_seed(args.seed)
np.random.seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
path = osp.join(osp.dirname(osp.realpath(__file__)), "..", "data")
data = Titanic(cached_dir=path)[0]
data.lazy_materialize()
target_col = data.target_col


#  collate function to batch raw DataFrame slices
def collate_td(index_batch: List[int]):
    x_batch = data.df.iloc[index_batch].reset_index(drop=True).drop(columns=[target_col])
    y_batch = data.get_label_ids(index_batch, device=device)
    return x_batch, y_batch


indices = np.arange(data.num_rows)
labels = data.df[target_col].values
train_idx, temp_idx = train_test_split(
    indices, test_size=0.3, stratify=labels, random_state=args.seed
)
val_idx, test_idx = train_test_split(
    temp_idx, test_size=2 / 3, stratify=labels[temp_idx], random_state=args.seed
)

# Replace get_dataloader with DataLoader over indices + collate_td
train_loader = DataLoader(train_idx.tolist(), batch_size=args.batch_size, shuffle=True, collate_fn=collate_td)
val_loader = DataLoader(val_idx.tolist(), batch_size=args.batch_size, shuffle=False, collate_fn=collate_td)
test_loader = DataLoader(test_idx.tolist(), batch_size=args.batch_size, shuffle=False, collate_fn=collate_td)

# Build model and optimizer
col_types = data.col_types
cat_cols = [c for c, t in col_types.items() if t == ColType.CATEGORICAL and c != target_col]
num_cols = [c for c, t in col_types.items() if t == ColType.NUMERICAL]
bin_cols = [c for c, t in col_types.items() if t == ColType.BINARY]
num_classes = data.num_classes

model = TransTabClassifier(
    categorical_columns=cat_cols,
    numerical_columns=num_cols,
    binary_columns=bin_cols,
    num_class=num_classes,
    hidden_dim=args.hidden_dim,
    num_layer=args.num_layers,
    num_attention_head=args.num_heads,
    hidden_dropout_prob=0.1,
    ffn_dim=args.hidden_dim * 2,
    activation="relu",
    device=device,
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)


def train(epoch: int) -> float:
    model.train()
    loss_accum = total_count = 0.0
    for x_batch, y in tqdm(train_loader, desc=f"Epoch: {epoch}"):
        logits, loss = model(x_batch, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_accum += loss.item() * y.size(0)
        total_count += y.size(0)
    return loss_accum / total_count


@torch.no_grad()
def test(loader: DataLoader) -> float:
    model.eval()
    ys, ps = [], []
    for x_batch, y in loader:
        logits, _ = model(x_batch)
        if num_classes <= 2:
            prob = torch.sigmoid(logits).view(-1)
        else:
            prob = torch.softmax(logits, dim=1)[:, 1]
        ys.append(y.cpu().numpy())
        ps.append(prob.cpu().numpy())
    return roc_auc_score(np.concatenate(ys), np.concatenate(ps))


metric = "AUC"
best_val_metric = test_metric = 0.0
times: List[float] = []

for epoch in range(1, args.epochs + 1):
    start = time.time()
    train_loss = train(epoch)
    train_auc = test(train_loader)
    val_auc = test(val_loader)
    tmp_test_auc = test(test_loader)

    if val_auc > best_val_metric:
        best_val_metric = val_auc
        test_metric = tmp_test_auc

    times.append(time.time() - start)
    print(
        f"Epoch [{epoch}/{args.epochs}]  "
        f"Train Loss: {train_loss:.4f}, "
        f"Train {metric}: {train_auc:.4f}, "
        f"Val {metric}: {val_auc:.4f}, "
        f"Test {metric}: {tmp_test_auc:.4f}"
    )

print(f"Mean time per epoch: {np.mean(times):.4f}s")
print(f"Total time: {sum(times):.4f}s")
print(f"Test {metric} at best Val: {test_metric:.4f}")
