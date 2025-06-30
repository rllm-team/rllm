# The Transtab method from the
# "TransTab: Learning Transferable Tabular Transformers Across Tables" paper.
# ArXiv: https://arxiv.org/abs/2205.09328

# Datasets  Titanic    Adult
# Acc       0.765      
# Time      40.7s      


import argparse
import sys
import time
from typing import Any, Dict, List
import os.path as osp

from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append("./")
sys.path.append("../")
from rllm.types import ColType
from rllm.datasets import Titanic
from rllm.nn.conv.table_conv import TransTabClassifier

parser = argparse.ArgumentParser()
parser.add_argument("--hidden_dim",   help="hidden dim",      type=int,   default=128)
parser.add_argument("--num_layers",                 type=int,   default=2)
parser.add_argument("--num_heads",                  type=int,   default=8)
parser.add_argument("--batch_size",                 type=int,   default=256)
parser.add_argument("--epochs",                     type=int,   default=50)
parser.add_argument("--lr",                         type=float, default=1e-3)
parser.add_argument("--wd",                         type=float, default=1e-4)
parser.add_argument("--seed",                       type=int,   default=42)
args = parser.parse_args()

# Set random seed and device
torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
data_path = osp.join(osp.dirname(osp.realpath(__file__)), "..", "data")
td = Titanic(cached_dir=data_path)[0]
df = td.df   # pandas.DataFrame
target_col = td.target_col

# Split dataset, here the ratio of train-val-test is 80%-10%-10%
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=args.seed, stratify=df[target_col]
)
train_df, val_df = train_test_split(
    train_df, test_size=0.125, random_state=args.seed, stratify=train_df[target_col]
)

# Dataset + collate_fn 
class DataFrameDataset(Dataset):
    def __init__(self, df: pd.DataFrame, target_col: str):
        self.df = df.reset_index(drop=True)
        self.target_col = target_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        y = row[self.target_col]
        x = row.drop(labels=[self.target_col])
        return x.to_frame().T, y

def collate_fn(batch):
    x_list, y_list = zip(*batch)
    batch_df = pd.concat(x_list, ignore_index=True)
    y = torch.tensor(y_list, dtype=torch.long)
    return batch_df, y

train_loader = DataLoader(
    DataFrameDataset(train_df, target_col),
    batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
)
val_loader = DataLoader(
    DataFrameDataset(val_df, target_col),
    batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
)
test_loader = DataLoader(
    DataFrameDataset(test_df, target_col),
    batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
)

# Column metadata
col_types = td.col_types
cat_cols = [c for c, t in col_types.items() if t == ColType.CATEGORICAL and c != target_col]
num_cols = [c for c, t in col_types.items() if t == ColType.NUMERICAL]
bin_cols = [c for c, t in col_types.items() if t == ColType.BINARY]

# Define model
model = TransTabClassifier(
    categorical_columns=cat_cols,
    numerical_columns=num_cols,
    binary_columns=bin_cols,
    num_class=td.num_classes,
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
    for batch_df, y in tqdm(train_loader, desc=f"Epoch: {epoch}"):
        y = y.to(device)
        logits, loss = model(batch_df, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_accum += loss.item() * y.size(0)
        total_count += y.size(0)

    return loss_accum / total_count

@torch.no_grad()
def test(loader: DataLoader) -> float:
    model.eval()
    correct = total = 0
    for batch_df, y in loader:
        y = y.to(device)
        logits = model(batch_df)
        preds = (torch.sigmoid(logits).view(-1) > 0.5).long()
        total += y.size(0)
        correct += (preds == y).sum().item()
    return correct / total

metric = "Acc"
best_val_metric = best_test_metric = 0.0
times = []

for epoch in range(1, args.epochs + 1):
    start = time.time()

    train_loss   = train(epoch)
    train_metric = test(train_loader)
    val_metric   = test(val_loader)
    test_metric  = test(test_loader)

    if val_metric > best_val_metric:
        best_val_metric  = val_metric
        best_test_metric = test_metric

    times.append(time.time() - start)
    print(
        f"Train Loss: {train_loss:.4f}, "
        f"Train {metric}: {train_metric:.4f}, "
        f"Val {metric}: {val_metric:.4f}, "
        f"Test {metric}: {test_metric:.4f}"
    )

print(f"\nMean time per epoch: {torch.tensor(times).mean():.4f}s")
print(f"Total time:               {sum(times):.4f}s")
print(
    f"Best Val {metric}: {best_val_metric:.4f}, "
    f"Best Test {metric}: {best_test_metric:.4f}"
)





'''
import argparse
import os
import time
from typing import List, Union, Optional #python -m examples.transtab

import pandas as pd
#pd.set_option('future.no_silent_downcasting', True)
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from rllm.types import ColType
from rllm.datasets import Adult, Titanic
from rllm.nn.conv.table_conv import TransTabClassifier

# ========== Argument parsing ==========
parser = argparse.ArgumentParser()
parser.add_argument("--hidden_dim",      type=int,   default=128)
parser.add_argument("--num_layers",      type=int,   default=2)
parser.add_argument("--num_heads",       type=int,   default=8)
parser.add_argument("--batch_size",      type=int,   default=256)
parser.add_argument("--epochs",          type=int,   default=20)
parser.add_argument("--lr",              type=float, default=1e-3)
parser.add_argument("--weight_decay",    type=float, default=1e-4)
parser.add_argument("--seed",            type=int,   default=42)
parser.add_argument("--cached_dir",      type=str,   default="data")
args = parser.parse_args()

# ========== Setup ==========
torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== Load and split Adult dataset ==========
# adult = Adult(cached_dir=args.cached_dir, forced_reload=False)[0]  # 返回 TableData
# df = adult.df  # pandas.DataFrame
# target_col = "income"

# # map string labels to 0/1
# df[target_col] = df[target_col].str.strip().map(lambda s: 1 if s.endswith(">50K") else 0)

titanic = Titanic(cached_dir=args.cached_dir, forced_reload=False)[0]
df = titanic.df
target_col = "Survived"



# train/val/test split: 80%/10%/10%
train_df, test_df = train_test_split(df, test_size=0.2, random_state=args.seed, stratify=df[target_col])
train_df, val_df  = train_test_split(train_df, test_size=0.125, random_state=args.seed, stratify=train_df[target_col])

# Extract column lists from rllm metadata
#col_types = adult.col_types  # dict[str, ColType]
col_types = titanic.col_types
cat_cols = [c for c,t in col_types.items() if t == ColType.CATEGORICAL and c != target_col]
num_cols = [c for c,t in col_types.items() if t == ColType.NUMERICAL]
bin_cols = [c for c,t in col_types.items() if t == ColType.BINARY]

# ========== Dataset & DataLoader ==========
class DataFrameDataset(Dataset):
    def __init__(self, df: pd.DataFrame, target_col: str):
        self.df = df.reset_index(drop=True)
        self.target_col = target_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        y = row[self.target_col]
        x = row.drop(labels=[self.target_col])
        # return as single-row DataFrame
        return x.to_frame().T, y

def collate_fn(batch):
    x_list, y_list = zip(*batch)
    batch_df = pd.concat(x_list, ignore_index=True)
    y = torch.tensor(y_list, dtype=torch.long)
    return batch_df, y

train_loader = DataLoader(
    DataFrameDataset(train_df, target_col),
    batch_size=args.batch_size, shuffle=True,
    collate_fn=collate_fn,
)
val_loader = DataLoader(
    DataFrameDataset(val_df, target_col),
    batch_size=args.batch_size, shuffle=False,
    collate_fn=collate_fn,
)
test_loader = DataLoader(
    DataFrameDataset(test_df, target_col),
    batch_size=args.batch_size, shuffle=False,
    collate_fn=collate_fn,
)

# ========== Model, optimizer, loss ==========
model = TransTabClassifier(
    categorical_columns=cat_cols,
    numerical_columns=num_cols,
    binary_columns=bin_cols,
    num_class=2,
    hidden_dim=args.hidden_dim,
    num_layer=args.num_layers,
    num_attention_head=args.num_heads,
    hidden_dropout_prob=0.1,
    ffn_dim=args.hidden_dim * 2,
    activation="relu",
    device=device,
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

# ========== Training and evaluation ==========

def train_one_epoch(epoch: int):
    model.train()
    total_loss = 0.0
    total_samples = 0
    for batch_df, y in train_loader:
        #batch_df, y = batch_df.to(device), y.to(device)
        y = y.to(device)
        logits, loss = model(batch_df, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y.size(0)
        total_samples += y.size(0)
    avg_loss = total_loss / total_samples
    print(f"[Epoch {epoch}] Train loss: {avg_loss:.4f}")
    return avg_loss

@torch.no_grad()
def evaluate(loader, split: str):
    model.eval()
    correct = 0
    total = 0
    for batch_df, y in loader:
        #batch_df, y = batch_df.to(device), y.to(device)
        y = y.to(device)
        logits = model(batch_df)  # no y → returns logits only
        preds = (torch.sigmoid(logits).view(-1) > 0.5).long()
        correct += (preds == y).sum().item()
        total += y.size(0)
    acc = correct / total
    print(f"{split} accuracy: {acc:.4f}")
    return acc

best_val_acc = 0.0
best_test_acc = 0.0
times = []

for epoch in range(1, args.epochs + 1):
    start = time.time()
    train_one_epoch(epoch)
    val_acc = evaluate(val_loader, "Validation")
    test_acc = evaluate(test_loader, "Test")
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_test_acc = test_acc
    times.append(time.time() - start)

print(f"Best val acc: {best_val_acc:.4f}, corresponding test acc: {best_test_acc:.4f}")
print(f"Average epoch time: {sum(times)/len(times):.2f}s")
'''