# The Transtab method from the
# "TransTab: Learning Transferable Tabular Transformers Across Tables" paper.
# ArXiv: https://arxiv.org/abs/2205.09328

# Datasets    Titanic                 Adult
#             pre_train  finetune     pre_train  finetune
# AUC(rept.)   -          -           0.88       0.90
# AUC(ours)   0.8157     0.7490       0.8350     0.8918
# Time        8.7s       9.5s         612.3s     807.6s

import argparse
import sys
import os.path as osp

import torch
from torch.utils.data import DataLoader

sys.path.append("./")
sys.path.append("../")
sys.path.append("../../")
from rllm.datasets import Titanic
from rllm.nn.models import TransTabClassifier
import utils


parser = argparse.ArgumentParser(description="TransTab Cross-Table (subsetA -> subsetB, 50% column overlap)")
parser.add_argument("--hidden_dim", type=int, default=128, help="Transformer hidden dim")
parser.add_argument("--num_layers", type=int, default=2, help="Number of transformer layers")
parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
parser.add_argument("--pre_epochs", type=int, default=100, help="Pre-train epochs on source table")
parser.add_argument("--finetune_epochs", type=int, default=100, help="Fine-tune epochs on target table")
parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
parser.add_argument("--wd", type=float, default=0, help="Weight decay")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--patience_pre", type=int, default=10, help="Early stopping patience (pre-train)")
parser.add_argument("--patience_ft", type=int, default=10, help="Early stopping patience (fine-tune)")
args = parser.parse_args()


utils.set_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data")
original_table = Titanic(cached_dir=path)[0]
target_column = original_table.target_col
utils.build_split_masks(original_table, target_col=target_column, seed=args.seed,
                        train_ratio=0.7, val_ratio=0.1, test_ratio=0.2)

# Construct two sub-tables with 50% column overlap ( pre-train, fine-tune)
source_table_cols, target_table_cols = utils.split_columns_half_overlap(
    original_table, target_col=target_column, seed=args.seed)
subtable_source = utils.TableView(original_table, keep_cols=source_table_cols, target_col=target_column)
subtable_target = utils.TableView(original_table, keep_cols=target_table_cols, target_col=target_column)

# Build loaders for subset source table (pre-train phase)
train_idx_source = utils.mask_to_index(subtable_source.train_mask)
val_idx_source = utils.mask_to_index(subtable_source.val_mask)
test_idx_source = (
    utils.mask_to_index(subtable_source.test_mask)
    if getattr(subtable_source, "test_mask", None) is not None
    else None
)

batch_pretrain = utils.make_batch_fn(subtable_source, target_column, device)
train_loader_source = DataLoader(
    train_idx_source.tolist(),
    batch_size=args.batch_size,
    shuffle=False,
    collate_fn=batch_pretrain,
)
val_loader_source = DataLoader(
    val_idx_source.tolist(),
    batch_size=args.batch_size,
    shuffle=False,
    collate_fn=batch_pretrain,
)
test_loader_source = (
    DataLoader(
        test_idx_source.tolist(),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=batch_pretrain,
    )
    if test_idx_source is not None
    else None
)

# Create model according to subset source table's schema
cat_source, num_source, bin_source, num_class_source = utils.get_column_partitions(subtable_source, target_column)
model = TransTabClassifier(
    categorical_columns=cat_source,
    numerical_columns=num_source,
    binary_columns=bin_source,
    num_class=num_class_source,
    hidden_dim=args.hidden_dim,
    num_layer=args.num_layers,
    num_attention_head=args.num_heads,
    ffn_dim=args.hidden_dim * 2,
    device=device,
).to(device)

# Run pre-training on source table
test_at_best_pre = utils.run_phase(
    phase_name="Pre-training",
    model=model,
    num_classes=num_class_source,
    train_loader=train_loader_source,
    val_loader=val_loader_source,
    optional_test_loader=test_loader_source,
    epochs=args.pre_epochs,
    patience=args.patience_pre,
    lr=args.lr,
    weight_decay=args.wd,
)
print("\n[Pre-training] Test @ Best Val on source table:")
print(f" AUC  {test_at_best_pre['auc']:.4f}")
print(f" Acc  {test_at_best_pre['acc']:.4f}")
print(f" F1m  {test_at_best_pre['f1_macro']:.4f}")

# Reconfigure model to subset target table's schema (update heads & column settings)
cat_target, num_target, bin_target, num_class_target = utils.get_column_partitions(subtable_target, target_column)
model.update({"cat": cat_target, "num": num_target, "bin": bin_target, "num_class": num_class_target})
model.to(device)

# Build loaders for subset target table (fine-tune phase)
train_idx_target = utils.mask_to_index(subtable_target.train_mask)
val_idx_target = utils.mask_to_index(subtable_target.val_mask)
test_idx_target = (
    utils.mask_to_index(subtable_target.test_mask)
    if getattr(subtable_target, "test_mask", None) is not None
    else None
)

batch_finetune = utils.make_batch_fn(subtable_target, target_column, device)
train_loader_target = DataLoader(
    train_idx_target.tolist(),
    batch_size=args.batch_size,
    shuffle=False,
    collate_fn=batch_finetune,
)
val_loader_target = DataLoader(
    val_idx_target.tolist(),
    batch_size=args.batch_size,
    shuffle=False,
    collate_fn=batch_finetune,
)
test_loader_target = (
    DataLoader(
        test_idx_target.tolist(),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=batch_finetune,
    )
    if test_idx_target is not None
    else None
)

# Run fine-tuning on target table
test_at_best_ft = utils.run_phase(
    phase_name="Fine-tuning",
    model=model,
    num_classes=num_class_target,
    train_loader=train_loader_target,
    val_loader=val_loader_target,
    optional_test_loader=test_loader_target,
    epochs=args.finetune_epochs,
    patience=args.patience_ft,
    lr=args.lr,
    weight_decay=args.wd,
)

print("\n[Final] Test @ Best Val:")
print(f" AUC  {test_at_best_ft['auc']:.4f}")
print(f" Acc  {test_at_best_ft['acc']:.4f}")
print(f" F1m  {test_at_best_ft['f1_macro']:.4f}")
