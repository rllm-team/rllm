# The Transtab method from the
# "TransTab: Learning Transferable Tabular Transformers Across Tables" paper.
# ArXiv: https://arxiv.org/abs/2205.09328

# Datasets    Titanic             Adult
#             set1       set2     set1       set2
# AUC(rept.)   -          -       0.88       0.90
# AUC(ours)   0.8142     0.6989   0.9004     0.8572
# Time        10.7s      3.5s     712.3s     806.9s

import argparse
import sys
import os.path as osp

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append("./")
sys.path.append("../")
sys.path.append("../../")
from rllm.datasets import Titanic
from rllm.nn.models import TransTabClassifier
import utils as U


parser = argparse.ArgumentParser(description="TransTab Cross-Table (subsetA -> subsetB, 50% column overlap)")
parser.add_argument("--hidden_dim", type=int, default=128, help="Transformer hidden dim")
parser.add_argument("--num_layers", type=int, default=2, help="Number of transformer layers")
parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
parser.add_argument("--pre_epochs", type=int, default=100, help="Pre-train epochs on subset A")
parser.add_argument("--finetune_epochs", type=int, default=100, help="Fine-tune epochs on subset B")
parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
parser.add_argument("--wd", type=float, default=0.0, help="Weight decay")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--patience_pre", type=int, default=10, help="Early stopping patience (pre-train)")
parser.add_argument("--patience_ft", type=int, default=10, help="Early stopping patience (fine-tune)")
args = parser.parse_args()


U.set_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data")
base_table = Titanic(cached_dir=path)[0]
target_column = base_table.target_col
U.build_split_masks(base_table, target_col=target_column, seed=args.seed,
                    train_ratio=0.7, val_ratio=0.1, test_ratio=0.2)

# Construct two sub-tables with 50% column overlap (A -> pre-train, B -> fine-tune)
A_cols, B_cols = U.split_columns_half_overlap(base_table, target_col=target_column, seed=args.seed)
subtable_A = U.SubTable(base_table, keep_cols=A_cols, target_col=target_column)
subtable_B = U.SubTable(base_table, keep_cols=B_cols, target_col=target_column)

# Build loaders for subset A (pre-train phase)
train_idxA = U.mask_to_index(subtable_A.train_mask)
val_idxA = U.mask_to_index(subtable_A.val_mask)
test_idxA = U.mask_to_index(subtable_A.test_mask) if getattr(subtable_A, "test_mask", None) is not None else None

collate_pretrain = U.build_collate_fn(subtable_A, target_column, device)
train_loaderA = DataLoader(train_idxA.tolist(), batch_size=args.batch_size, shuffle=True, collate_fn=collate_pretrain)
val_loaderA = DataLoader(val_idxA.tolist(), batch_size=args.batch_size, shuffle=False, collate_fn=collate_pretrain)
test_loaderA = (DataLoader(test_idxA.tolist(), batch_size=args.batch_size, shuffle=False, collate_fn=collate_pretrain)
                if test_idxA is not None else None)

# Create model according to subset A's schema
catA, numA, binA, nclsA = U.get_column_partitions(subtable_A, target_column)
model = TransTabClassifier(
    categorical_columns=catA,
    numerical_columns=numA,
    binary_columns=binA,
    num_class=nclsA,
    hidden_dim=args.hidden_dim,
    num_layer=args.num_layers,
    num_attention_head=args.num_heads,
    ffn_dim=args.hidden_dim * 2,
).to(device)

# Run pre-training on subset A
test_at_best_pre = U.run_phase(
    phase_name="Pre",
    model=model,
    num_classes=nclsA,
    train_loader=train_loaderA,
    val_loader=val_loaderA,
    optional_test_loader=test_loaderA,
    epochs=args.pre_epochs,
    patience=args.patience_pre,
    lr=args.lr,
    weight_decay=args.wd,
)
print("\n[Pre ] Test @ Best Val on subset A:")
print(f" AUC  {test_at_best_pre['auc']:.4f}")
print(f" Acc  {test_at_best_pre['acc']:.4f}")
print(f" F1m  {test_at_best_pre['f1_macro']:.4f}")

# Reconfigure model to subset B's schema (update heads & column settings)
catB, numB, binB, nclsB = U.get_column_partitions(subtable_B, target_column)
model.update({"cat": catB, "num": numB, "bin": binB, "num_class": nclsB})
model.to(device)

# Build loaders for subset B (fine-tune phase)
train_idxB = U.mask_to_index(subtable_B.train_mask)
val_idxB = U.mask_to_index(subtable_B.val_mask)
test_idxB = U.mask_to_index(subtable_B.test_mask) if getattr(subtable_B, "test_mask", None) is not None else None

collate_finetune = U.build_collate_fn(subtable_B, target_column, device)
train_loaderB = DataLoader(train_idxB.tolist(), batch_size=args.batch_size, shuffle=True, collate_fn=collate_finetune)
val_loaderB = DataLoader(val_idxB.tolist(), batch_size=args.batch_size, shuffle=False, collate_fn=collate_finetune)
test_loaderB = (DataLoader(test_idxB.tolist(), batch_size=args.batch_size, shuffle=False, collate_fn=collate_finetune)
                if test_idxB is not None else None)

# Run fine-tuning on subset B
test_at_best_ft = U.run_phase(
    phase_name="FT ",
    model=model,
    num_classes=nclsB,
    train_loader=train_loaderB,
    val_loader=val_loaderB,
    optional_test_loader=test_loaderB,
    epochs=args.finetune_epochs,
    patience=args.patience_ft,
    lr=args.lr,
    weight_decay=args.wd,
)

# 11) Final evaluation after restoring best FT weights
final_test = (
    U.evaluate(model, test_loaderB, nclsB)
    if test_loaderB is not None
    else {"auc": np.nan, "acc": np.nan, "f1_macro": np.nan}
)

print("\n[Final] Test @ Best Val (tracked during FT):")
print(f" AUC  {test_at_best_ft['auc']:.4f}")
print(f" Acc  {test_at_best_ft['acc']:.4f}")
print(f" F1m  {test_at_best_ft['f1_macro']:.4f}")

print("\n[Final] Test after loading best FT weights:")
print(f" AUC  {final_test['auc']:.4f}")
print(f" Acc  {final_test['acc']:.4f}")
print(f" F1m  {final_test['f1_macro']:.4f}")
