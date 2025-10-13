from typing import List, Dict, Tuple, Optional
import time
import os
import random

from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from rllm.types import ColType


__all__ = [
    # Core: data/columns & views
    "get_column_partitions",
    "split_columns_half_overlap",
    "TableView",
    # Dataset splits & loaders
    "build_split_masks",
    "make_batch_fn",
    "mask_to_index",
    # Training utilities
    "set_seed", 
    "EarlyStopping",
    "evaluate",
    "train_epoch",
    "run_phase",
]


def set_seed(seed: int):
    """Set random seed for reproducibility across runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

def mask_to_index(mask_tensor: torch.Tensor) -> np.ndarray:
    r"""Convert boolean/0-1 mask tensor to numpy index array."""
    m = mask_tensor.cpu().numpy()
    if m.dtype == np.bool_:
        return np.where(m)[0]
    return np.where(m != 0)[0]


def make_batch_fn(table, target_col: str, device: torch.device):
    r"""Build a collate_fn for DataLoader: indices -> (X_batch, y_batch)."""
    def _collate(index_batch: List[int]):
        x_batch = table.df.iloc[index_batch].reset_index(drop=True).drop(columns=[target_col])
        y_batch = table.get_label_ids(index_batch, device=device)
        return x_batch, y_batch
    return _collate


def get_column_partitions(table, target_col: str) -> Tuple[List[str], List[str], List[str], int]:
    r"""Return categorical, numerical, binary columns and number of classes."""
    col_types = table.col_types
    cat_cols = [c for c, t in col_types.items() if t == ColType.CATEGORICAL and c != target_col]
    num_cols = [c for c, t in col_types.items() if t == ColType.NUMERICAL]
    bin_cols = [c for c, t in col_types.items() if t == ColType.BINARY]
    num_classes = table.num_classes
    return cat_cols, num_cols, bin_cols, num_classes


class EarlyStopping:
    r"""
    Minimal early stopping with configurable direction.
    Args:
        patience: epochs to wait without improvement.
        min_delta: minimal absolute improvement to reset patience.
        mode: "max" (e.g., AUC/Acc/F1) or "min" (e.g., loss).
        metric_name: for logging only.
        restore_best: save & allow restore() of best weights.
    """
    def __init__(self,
                 patience: int = 5,
                 min_delta: float = 1e-5,
                 mode: str = "max",
                 metric_name: str = "val_metric",
                 restore_best: bool = True):
        assert mode in ("max", "min")
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.mode = mode
        self.metric_name = metric_name
        self.restore_best = restore_best

        self.best = -np.inf if mode == "max" else np.inf
        self.counter = 0
        self.best_value: Optional[float] = None
        self.best_state: Optional[Dict[str, torch.Tensor]] = None

    def step(self, value: float, model: torch.nn.Module) -> bool:
        """Return True if training should stop."""
        improved = (value > self.best + self.min_delta) if self.mode == "max" \
            else (value < self.best - self.min_delta)
        if improved:
            self.best = float(value)
            self.best_value = float(value)
            self.counter = 0
            if self.restore_best:
                self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

    def restore(self, model: torch.nn.Module) -> None:
        """Load best weights if saved."""
        if self.restore_best and self.best_state is not None:
            model.load_state_dict(self.best_state)


def split_features_one_type(feats: List[str], rng: np.random.RandomState) -> Tuple[List[str], List[str], List[str]]:
    r"""Split features into overlap, set A, and set B (~1/3 overlap)."""
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
    seed: int = 42,
) -> Tuple[List[str], List[str]]:
    r"""Split columns into two sets with 50% overlap (for transfer experiments)."""
    rng = np.random.RandomState(seed)
    col_types = table.col_types

    cats = [c for c, t in col_types.items() if t == ColType.CATEGORICAL and c != target_col]
    nums = [c for c, t in col_types.items() if t == ColType.NUMERICAL]
    bins = [c for c, t in col_types.items() if t == ColType.BINARY]

    O_c, A_c, B_c = split_features_one_type(cats, rng)
    O_n, A_n, B_n = split_features_one_type(nums, rng)
    O_b, A_b, B_b = split_features_one_type(bins, rng)

    overlap = O_c + O_n + O_b
    subset1_only = A_c + A_n + A_b
    subset2_only = B_c + B_n + B_b

    cols_subset1 = overlap + subset1_only
    cols_subset2 = overlap + subset2_only

    def _dedup(seq: List[str]) -> List[str]:
        seen, out = set(), []
        for x in seq:
            if x not in seen:
                out.append(x)
                seen.add(x)
        return out

    return _dedup(cols_subset1), _dedup(cols_subset2)


class TableView:
    r"""Wrapper for creating a sub-table with selected columns and masks."""
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
    r"""Ensure dataset has train/val/test split masks; create them if absent."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "ratios must sum to 1"

    has_train = getattr(table, "train_mask", None) is not None
    has_val = getattr(table, "val_mask", None) is not None
    has_test = getattr(table, "test_mask", None) is not None

    if has_train and has_val and has_test:
        return

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
    tr_mask = np.zeros(n, dtype=bool)
    tr_mask[train_idx] = True
    va_mask = np.zeros(n, dtype=bool)
    va_mask[val_idx] = True
    te_mask = np.zeros(n, dtype=bool)
    te_mask[test_idx] = True

    table.train_mask = torch.tensor(tr_mask, dtype=torch.bool)
    table.val_mask = torch.tensor(va_mask, dtype=torch.bool)
    table.test_mask = torch.tensor(te_mask, dtype=torch.bool)


@torch.no_grad()
def evaluate(model, loader: DataLoader, num_classes: int) -> Dict[str, float]:
    r"""Evaluate AUC / Accuracy / Macro-F1 on a given loader."""
    model.eval()
    labels_list, probs_list, preds_list = [], [], []
    for x_batch, y in loader:
        logits, _ = model(x_batch)
        if num_classes <= 2:
            prob = torch.sigmoid(logits).view(-1, 1)          # [N,1]
            prob = torch.cat([1 - prob, prob], dim=1)         # [N,2]
        else:
            prob = F.softmax(logits, dim=1)                   # [N,C]
        pred = prob.argmax(dim=1)

        labels_list.append(y.cpu().numpy())
        probs_list.append(prob.cpu().numpy())
        preds_list.append(pred.cpu().numpy())

    y_true = np.concatenate(labels_list, axis=0)
    y_prob = np.concatenate(probs_list, axis=0)
    y_pred = np.concatenate(preds_list, axis=0)

    try:
        auc = (
            roc_auc_score(y_true, y_prob[:, 1])
            if y_prob.shape[1] == 2
            else roc_auc_score(y_true, y_prob, multi_class="ovr")
        )
    except Exception:
        auc = float("nan")

    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return {"auc": auc, "acc": acc, "f1_macro": f1m}


def train_epoch(model, loader: DataLoader, optimizer: torch.optim.Optimizer) -> float:
    r"""Train the model for one epoch and return the average loss."""
    model.train()
    loss_sum, count = 0.0, 0
    for x_batch, y in tqdm(loader, leave=False, desc="Train"):
        _, loss = model(x_batch, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * y.size(0)
        count += y.size(0)
    return loss_sum / max(count, 1)


def run_phase(
    phase_name: str,
    model,
    num_classes: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optional_test_loader: Optional[DataLoader],
    epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
):
    r"""Run a training phase (pre-train or fine-tune) with early stopping and optional test tracking."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    es = EarlyStopping(patience=patience, mode="max", metric_name="val_auc")

    best_val_auc = -np.inf
    test_at_best = {"auc": float("-inf"), "acc": float("-inf"), "f1_macro": float("-inf")}

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer)
        metrics_train = evaluate(model, train_loader, num_classes)
        metrics_val = evaluate(model, val_loader, num_classes)
        metrics_test = (
            {"auc": np.nan, "acc": np.nan, "f1_macro": np.nan}
            if optional_test_loader is None else
            evaluate(model, optional_test_loader, num_classes)
        )
        t = time.time() - t0

        if metrics_val["auc"] > best_val_auc:
            best_val_auc = metrics_val["auc"]
            test_at_best = metrics_test.copy()

        print(
            f"[{phase_name}] Epoch {epoch:02d}/{epochs}  "
            f"Loss {train_loss:.4f}  "
            f"Train AUC {metrics_train['auc']:.4f} Val AUC {metrics_val['auc']:.4f} Test AUC {metrics_test['auc']:.4f} "
            f"Val Acc {metrics_val['acc']:.4f} Test Acc {metrics_test['acc']:.4f}  "
            f"Val F1 {metrics_val['f1_macro']:.4f} Test F1 {metrics_test['f1_macro']:.4f} ({t:.2f}s)"
        )

        if es.step(metrics_val["auc"], model):
            print(f"[{phase_name}] Early stopping triggered (patience={patience}).")
            break

    # Restore best weights for this phase
    es.restore(model)

    return test_at_best
