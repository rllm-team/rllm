from typing import List, Tuple

import numpy as np
from numpy.random import Generator
import torch
from sklearn.model_selection import train_test_split

from rllm.types import ColType
from rllm.data import TableData


__all__ = [
    # Core: data/columns & subtables
    "get_column_partitions",
    "split_columns_half_overlap",
    "create_subtable",
    # Dataset splits & loaders
    "build_split_masks",
    "mask_to_index",
]


def get_column_partitions(table, target_col: str) -> Tuple[List[str], List[str], List[str], int]:
    # Partition table columns by ColType, returning categorical, numerical, binary columns and number of classes.
    # Note: In TransTab's terminology, "categorical" corresponds to TEXT in TableData (merged tokenized features)
    # CRITICAL: Exclude target_col from all column lists to match feat_dict and colname_token_ids
    col_types = table.col_types
    cat_cols = [c for c, t in col_types.items() if t == ColType.TEXT and c != target_col]
    num_cols = [c for c, t in col_types.items() if t == ColType.NUMERICAL and c != target_col]
    bin_cols = [c for c, t in col_types.items() if t == ColType.BINARY and c != target_col]
    num_classes = table.num_classes
    return cat_cols, num_cols, bin_cols, num_classes


def split_features(
    feat_cols: List[str],
    rng: Generator,
    overlap_ratio: float = 1 / 3,
) -> Tuple[List[str], List[str], List[str]]:
    # Randomly split all feature columns into (overlap, set_a, set_b).
    cols = feat_cols.copy()
    rng.shuffle(cols)
    n = len(cols)
    n_overlap = int(round(n * overlap_ratio))
    rest = n - n_overlap
    n_a = rest // 2

    overlap = cols[:n_overlap]
    set_a = cols[n_overlap:n_overlap + n_a]
    set_b = cols[n_overlap + n_a:]
    return overlap, set_a, set_b


def split_columns_half_overlap(
    table,
    target_col: str,
    rng: Generator,
) -> Tuple[List[str], List[str]]:
    # Split columns into two sets with 50% overlap (for transfer experiments).
    all_feat_cols = [c for c in table.col_types.keys() if c != target_col]

    overlap, set_a, set_b = split_features(all_feat_cols, rng=rng)
    cols_subset_a = overlap + set_a
    cols_subset_b = overlap + set_b

    # optional: fail-fast uniqueness checks
    assert len(cols_subset_a) == len(set(cols_subset_a))
    assert len(cols_subset_b) == len(set(cols_subset_b))

    return cols_subset_a, cols_subset_b


def create_subtable(base_table, keep_cols: List[str], target_col: str, tokenizer_config=None):
    """
    Create a new TableData from a subset of columns of the base table.
    This is better than TableView because:
    1. Creates a real TableData object with proper feat_dict and colname_token_ids generation
    2. Ensures complete consistency with single-table processing
    3. No need to manually manage column filtering or slicing
    Args:
        base_table: The original TableData
        keep_cols: List of columns to keep (including target_col)
        target_col: Name of the target column
        tokenizer_config: TokenizerConfig for text processing (should be same as base_table)
    Returns:
        A new TableData object with only the specified columns
    """
    # Ensure column order: features first, then target
    cols = [c for c in keep_cols if c != target_col] + [target_col]
    # Extract subset DataFrame
    sub_df = base_table.df[cols].copy()
    # Extract subset col_types (maintaining order from cols)
    sub_col_types = {c: base_table.col_types[c] for c in cols if c in base_table.col_types}
    # Create new TableData - this will regenerate feat_dict and colname_token_ids correctly
    subtable = TableData(
        df=sub_df,
        col_types=sub_col_types,
        target_col=target_col,
        tokenizer_config=tokenizer_config,
        categorical_as_text=(tokenizer_config is not None),
    )
    if hasattr(base_table, 'train_mask'):
        subtable.train_mask = base_table.train_mask
    if hasattr(base_table, 'val_mask'):
        subtable.val_mask = base_table.val_mask
    if hasattr(base_table, 'test_mask'):
        subtable.test_mask = base_table.test_mask
    return subtable


def build_split_masks(
    table,
    target_col: str,
    seed: bool = None,
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
