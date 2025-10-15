from typing import List, Tuple

import numpy as np
from numpy.random import Generator
import torch
from sklearn.model_selection import train_test_split

from rllm.types import ColType


__all__ = [
    # Core: data/columns & views
    "get_column_partitions",
    "split_columns_half_overlap",
    "TableView",
    # Dataset splits & loaders
    "build_split_masks",
    "mask_to_index",
]


def get_column_partitions(table, target_col: str) -> Tuple[List[str], List[str], List[str], int]:
    # Partition table columns by ColType, returning categorical, numerical, binary columns and number of classes.
    col_types = table.col_types
    cat_cols = [c for c, t in col_types.items() if t == ColType.CATEGORICAL and c != target_col]
    num_cols = [c for c, t in col_types.items() if t == ColType.NUMERICAL]
    bin_cols = [c for c, t in col_types.items() if t == ColType.BINARY]
    num_classes = table.num_classes
    return cat_cols, num_cols, bin_cols, num_classes


def split_features_one_type(feats: List[str], rng: Generator) -> Tuple[List[str], List[str], List[str]]:
    # Split one type of features into overlap, set A, and set B (~1/3 overlap).
    feats = feats.copy()
    rng.shuffle(feats)
    n = len(feats)
    n_overlap = int(round(n / 3.0))
    rest = n - n_overlap
    n_aonly = int(round(rest / 2.0))
    overlap = feats[:n_overlap]
    set_a = feats[n_overlap:n_overlap + n_aonly]
    set_b = feats[n_overlap + n_aonly:]
    return overlap, set_a, set_b


def split_columns_half_overlap(
    table,
    target_col: str,
    rng: Generator,
) -> Tuple[List[str], List[str]]:
    # Split columns into two sets with 50% overlap (for transfer experiments).
    col_types = table.col_types

    cats = [c for c, t in col_types.items() if t == ColType.CATEGORICAL and c != target_col]
    nums = [c for c, t in col_types.items() if t == ColType.NUMERICAL and c != target_col]
    bins = [c for c, t in col_types.items() if t == ColType.BINARY and c != target_col]

    overlap_cat, subset_a_cat, subset_b_cat = split_features_one_type(cats, rng)
    overlap_num, subset_a_num, subset_b_num = split_features_one_type(nums, rng)
    overlap_bin, subset_a_bin, subset_b_bin = split_features_one_type(bins, rng)

    overlap = overlap_cat + overlap_num + overlap_bin
    subset_a = subset_a_cat + subset_a_num + subset_a_bin
    subset_b = subset_b_cat + subset_b_num + subset_b_bin

    cols_subset_a = overlap + subset_a
    cols_subset_b = overlap + subset_b

    return cols_subset_a, cols_subset_b


class TableView:
    # Create a subtable view from the base table, keeping selected columns and masks.
    def __init__(self, base_table, keep_cols: List[str], target_col: str):
        self.base_table = base_table
        self.target_col = target_col

        cols = [c for c in keep_cols if c != target_col] + [target_col]
        self.df = base_table.df[cols].copy()

        self.col_types = {c: t for c, t in base_table.col_types.items() if c in self.df.columns}
        self.num_classes = base_table.num_classes

        self.train_mask = getattr(base_table, "train_mask", None)
        self.val_mask = getattr(base_table, "val_mask", None)
        self.test_mask = getattr(base_table, "test_mask", None)

        self.get_label_ids = base_table.get_label_ids


def build_split_masks(
    table,
    target_col: str,
    seed: int = 42,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    stratify: bool = True,
):
    # Generate train/val/test masks by given ratios with optional stratification and assign them to the table.
    n = len(table.df)
    idx_all = np.arange(n)
    y_raw = table.df[target_col].values
    strat = y_raw if stratify else None
    temp_ratio = val_ratio + test_ratio
    train_idx, temp_idx = train_test_split(
        idx_all,
        test_size=temp_ratio,
        random_state=seed,
        shuffle=True,
        stratify=strat,
    )
    strat_temp = y_raw[temp_idx] if stratify else None
    test_frac_in_temp = test_ratio / temp_ratio
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=test_frac_in_temp,
        random_state=seed,
        shuffle=True,
        stratify=strat_temp,
    )
    train_mask = np.zeros(n, dtype=bool)
    train_mask[train_idx] = True
    val_mask = np.zeros(n, dtype=bool)
    val_mask[val_idx] = True
    test_mask = np.zeros(n, dtype=bool)
    test_mask[test_idx] = True

    table.train_mask = torch.tensor(train_mask, dtype=torch.bool)
    table.val_mask = torch.tensor(val_mask, dtype=torch.bool)
    table.test_mask = torch.tensor(test_mask, dtype=torch.bool)


def mask_to_index(mask_tensor: torch.Tensor) -> np.ndarray:
    # Convert boolean/0-1 mask tensor to numpy index array.
    mask = mask_tensor.cpu().numpy()
    if mask.dtype == np.bool_:
        return np.where(mask)[0]
    return np.where(mask != 0)[0]
