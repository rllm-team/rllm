import os
import random
import time
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


__all__ = [
    "set_seed",
    "make_batch_fn",
    "EarlyStopping",
    "evaluate",
    "train_epoch",
    "run_phase",
]


def set_seed(seed: int):
    # Set random seed for reproducibility across runs.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def make_batch_fn(table, target_col: str, device: torch.device, use_tabledata: bool = True):
    """
    Build a collate_fn for DataLoader: indices -> (X_batch, y_batch).

    Args:
        table: TableData object
        target_col: Name of target column
        device: Device to place tensors on
        use_tabledata: If True, return TableData slice instead of DataFrame
    """
    if use_tabledata:
        # New TableData-based collate function
        def _collate(index_batch: List[int]):
            indices = torch.tensor(index_batch, dtype=torch.long)
            # Use TableData's tensor slicing to get a subset
            x_batch = table[indices]  # Returns a new TableData with sliced feat_dict
            y_batch = table.y[index_batch]
            return x_batch, y_batch
        return _collate
    else:
        # Original DataFrame-based collate function
        def _collate(index_batch: List[int]):
            x_batch = table.df.iloc[index_batch].reset_index(drop=True).drop(columns=[target_col])
            y_batch = table.y[index_batch]
            return x_batch, y_batch
        return _collate


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
        # Return True if training should stop.
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
        # Load best weights if saved.
        if self.restore_best and self.best_state is not None:
            model.load_state_dict(self.best_state)


@torch.no_grad()
def evaluate(model, loader: DataLoader, num_classes: int) -> Dict[str, float]:
    # Evaluate AUC / Accuracy / Macro-F1 on a given loader.
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


def train_epoch(model, loader: DataLoader, optimizer: torch.optim.Optimizer, max_grad_norm: float = 1.0) -> float:
    model.train()
    loss_sum, count = 0.0, 0
    for x_batch, y in tqdm(loader, leave=False, desc="Train"):
        _, loss = model(x_batch, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # Gradient clipping to prevent gradient explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
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
    # Run a training phase (pre-train or fine-tune) with early stopping and optional test tracking.
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    early_stopper = EarlyStopping(patience=patience, mode="max", metric_name="val_auc")

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

        if early_stopper.step(metrics_val["auc"], model):
            print(f"[{phase_name}] Early stopping triggered (patience={patience}).")
            break

    # Restore best weights for this phase
    early_stopper.restore(model)

    return test_at_best
