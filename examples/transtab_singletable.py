# The Transtab method from the
# "TransTab: Learning Transferable Tabular Transformers Across Tables" paper.
# ArXiv: https://arxiv.org/abs/2205.09328    

import argparse
import sys
import time
from typing import Any, Dict, List
import os.path as osp
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.preprocessing import LabelEncoder
# REMOVED: No longer needed as we use pre-defined masks
# from sklearn.model_selection import train_test_split

sys.path.append("./")
sys.path.append("../")
from rllm.types import ColType
# CHANGED: Import your custom dataset
from rllm.datasets import MSTrafficMarylandDataset
from rllm.nn.models import TransTabClassifier

parser = argparse.ArgumentParser()
parser.add_argument("--hidden_dim",   help="hidden dim",      type=int,   default=128)
parser.add_argument("--num_layers",                           type=int,   default=2)
parser.add_argument("--num_heads",                            type=int,   default=8)
parser.add_argument("--batch_size",                           type=int,   default=64)
parser.add_argument("--epochs",                               type=int,   default=20)
parser.add_argument("--lr",                                   type=float, default=1e-3)
parser.add_argument("--wd",                                   type=float, default=1e-4)
parser.add_argument("--seed",                                 type=int,   default=42)
args = parser.parse_args()

# Set random seed and device
torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== CHANGED: Load dataset and use pre-defined masks ==========
print("Loading MSTrafficDataset...")
data_path = osp.join(osp.dirname(osp.realpath(__file__)), "..", "data")
# Load your MSTrafficDataset
td = MSTrafficMarylandDataset(cached_dir=data_path)[0]
df = td.df  # The full pandas.DataFrame
target_col = td.target_col
label_encoder = LabelEncoder()
df[target_col] = label_encoder.fit_transform(df[target_col])

# Use the train/val/test masks that were loaded with the dataset
# This ensures a fixed, reproducible split.
train_df = df.loc[td.train_mask.numpy()]
val_df = df.loc[td.val_mask.numpy()]
test_df = df.loc[td.test_mask.numpy()]

print(f"Dataset loaded. Target: '{target_col}'. Num classes: {td.num_classes}")
print(f"Split sizes: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")


# Dataset + collate_fn (No changes needed here)
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
    # The target values should already be integer-encoded by your dataset processor
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


# Column metadata (No changes needed here)
col_types = td.col_types
cat_cols = [c for c, t in col_types.items() if t == ColType.CATEGORICAL and c != target_col]
num_cols = [c for c, t in col_types.items() if t == ColType.NUMERICAL]
bin_cols = [c for c, t in col_types.items() if t == ColType.BINARY]

# Define model (No changes needed, `num_class` is now automatically multi-class)
print("Initializing TransTabClassifier model...")
model = TransTabClassifier(
    categorical_columns=cat_cols,
    numerical_columns=num_cols,
    binary_columns=bin_cols,
    num_class=td.num_classes, # This will now be > 2
    hidden_dim=args.hidden_dim,
    num_layer=args.num_layers,
    num_attention_head=args.num_heads,
    hidden_dropout_prob=0.1,
    ffn_dim=args.hidden_dim * 2,
    activation="relu",
    device=device,
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

# train function (No changes needed here)
def train(epoch: int) -> float:
    model.train()
    loss_accum = total_count = 0.0
    for batch_df, y in tqdm(train_loader, desc=f"Epoch: {epoch}"):
        y = y.to(device)
        # The model internally uses CrossEntropyLoss for multi-class
        logits, loss = model(batch_df, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_accum += loss.item() * y.size(0)
        total_count += y.size(0)

    return loss_accum / total_count

# test function (No changes needed, it already handles multi-class)
@torch.no_grad()
def test(loader: DataLoader) -> float:
    model.eval()
    correct = total = 0
    for batch_df, y in loader:
        y = y.to(device)
        out = model(batch_df)
        logits = out[0] if isinstance(out, tuple) else out
        
        # This logic correctly switches to argmax for multi-class
        if model.num_class > 2:
            preds = logits.argmax(dim=1)
        else: # For binary cases
            preds = (torch.sigmoid(logits).view(-1) > 0.5).long()
        
        total += y.size(0)
        correct += (preds == y).sum().item()
    return correct / total

# Main training loop (No changes needed here)
metric = "Acc"
best_val_metric = best_test_metric = 0.0
times = []

print("Starting training...")
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

print(f"\nTraining finished.")
print(f"Mean time per epoch: {torch.tensor(times).mean():.4f}s")
print(f"Total time:                       {sum(times):.4f}s")
print(
    f"Best Val {metric}: {best_val_metric:.4f}, "
    f"Corresponding Test {metric}: {best_test_metric:.4f}"
)
