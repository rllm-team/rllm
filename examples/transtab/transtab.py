# The Transtab method from the
# "TransTab: Learning Transferable Tabular Transformers Across Tables" paper.
# ArXiv: https://arxiv.org/abs/2205.09328
# This is a single-table learning example.

# Datasets     Titanic    Adult
# AUC(rept.)    -         0.90
# AUC(ours)    0.868      0.894
# Time         27.7s      2236.0s

import argparse
import sys
import time
from typing import List
import os.path as osp

import numpy as np
from transformers import BertTokenizerFast
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

sys.path.append("./")
sys.path.append("../")
sys.path.append("../../")
from rllm.types import ColType
from rllm.datasets import Titanic, Adult
from rllm.nn.models import TransTabClassifier
from rllm.preprocessing import TokenizerConfig
import utils_run

parser = argparse.ArgumentParser()
parser.add_argument("--hidden_dim", type=int, default=128, help="Transformer hidden dim")
parser.add_argument("--num_layers", type=int, default=2, help="Number of transformer layers")
parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
parser.add_argument("--wd", type=float, default=5e-4, help="Weight decay")
parser.add_argument("--seed", type=int, default=7, help="Random seed")
parser.add_argument("--dataset", type=str, default="titanic", choices=["titanic", "adult"])
parser.add_argument("--tokenizer_dir", type=str, default="./tokenizer", help="Tokenizer directory")
args = parser.parse_args()

# Set random seed and device
# utils_run.set_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data")

# Initialize tokenizer for TableData preprocessing
tokenizer = BertTokenizerFast.from_pretrained(
    args.tokenizer_dir if osp.exists(args.tokenizer_dir) else "bert-base-uncased")
if not osp.exists(args.tokenizer_dir):
    tokenizer.save_pretrained(args.tokenizer_dir)

# Create tokenizer config for TableData
tokenizer_config = TokenizerConfig(
    tokenizer=tokenizer,
    pad_token_id=tokenizer.pad_token_id,
    tokenize_combine=True,
    include_colname=True,
    save_colname_token_ids=True,
)

# Load dataset with tokenization (use cache for reproducibility)
if args.dataset == "titanic":
    data = Titanic(cached_dir=path, tokenizer_config=tokenizer_config, forced_reload=True)[0]
elif args.dataset == "adult":
    data = Adult(cached_dir=path, tokenizer_config=tokenizer_config, forced_reload=True)[0]
target_col = data.target_col

# Create train/val/test splits
indices = np.arange(data.num_rows)
labels = data.y.numpy()
train_idx, temp_idx = train_test_split(
    indices, test_size=0.3, stratify=labels, random_state=args.seed
)
val_idx, test_idx = train_test_split(
    temp_idx, test_size=2 / 3, stratify=labels[temp_idx], random_state=args.seed
)

# Create DataLoader with TableData-based collate function
batch_fn = utils_run.make_batch_fn(data, target_col, device, use_tabledata=True)
train_loader = DataLoader(train_idx.tolist(), batch_size=args.batch_size, shuffle=False, collate_fn=batch_fn)
val_loader = DataLoader(val_idx.tolist(), batch_size=args.batch_size, shuffle=False, collate_fn=batch_fn)
test_loader = DataLoader(test_idx.tolist(), batch_size=args.batch_size, shuffle=False, collate_fn=batch_fn)

# Build model and optimizer
col_types = data.col_types
cat_cols = [c for c, t in col_types.items() if t == ColType.TEXT and c != target_col]
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
    ffn_dim=args.hidden_dim * 2,
    device=device,
    tokenizer=tokenizer,  # Pass the same tokenizer instance used in TableData
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

metric = "AUC"
best_val_metric = test_metric = 0.0
times: List[float] = []

for epoch in range(1, args.epochs + 1):
    start = time.time()
    train_loss = utils_run.train_epoch(model, train_loader, optimizer)
    train_auc = utils_run.evaluate(model, train_loader, num_classes)["auc"]
    val_auc = utils_run.evaluate(model, val_loader, num_classes)["auc"]
    tmp_test_auc = utils_run.evaluate(model, test_loader, num_classes)["auc"]

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
