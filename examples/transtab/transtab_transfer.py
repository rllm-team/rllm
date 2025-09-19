# The Transtab method from the
# "TransTab: Learning Transferable Tabular Transformers Across Tables" paper.
# ArXiv: https://arxiv.org/abs/2205.09328

# Datasets  Adult
#           set1       set2
# AUC       0.9004     0.8572

import argparse
import time
from typing import List, Tuple, Dict

from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from rllm.types import ColType
from rllm.datasets import Adult
from rllm.nn.models import TransTabClassifier


parser = argparse.ArgumentParser(description="TransTab Cross-Table (subsetA -> subsetB, 50% col overlap)")
parser.add_argument("--cached_dir", type=str, default="./data", help="Root directory for datasets")
parser.add_argument("--hidden_dim", type=int, default=128, help="Transformer hidden dim")
parser.add_argument("--num_layers", type=int, default=2, help="Number of transformer layers")
parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
parser.add_argument("--pre_epochs", type=int, default=100, help="Pre-train epochs on subset-A")
parser.add_argument("--finetune_epochs", type=int, default=100, help="Fine-tune epochs on subset-B")
parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
parser.add_argument("--wd", type=float, default=0, help="Weight decay")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--patience_pre", type=int, default=10, help="Early stopping patience for pre-training")
parser.add_argument("--patience_ft", type=int, default=10, help="Early stopping patience for fine-tuning")
args = parser.parse_args()


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mask_to_index(mask_tensor: torch.Tensor) -> np.ndarray:
    m = mask_tensor.cpu().numpy()
    if m.dtype == np.bool_:
        return np.where(m)[0]
    return np.where(m != 0)[0]


def make_collate_td(table, target_col: str, device: torch.device):
    def _collate(index_batch: List[int]):
        x_batch = table.df.iloc[index_batch].reset_index(drop=True).drop(columns=[target_col])
        y_batch = table.get_label_ids(index_batch, device=device)
        return x_batch, y_batch
    return _collate


@torch.no_grad()
def evaluate(model: TransTabClassifier, loader: DataLoader, num_classes: int) -> Dict[str, float]:
    model.eval()
    ys, probs_list, preds_list = [], [], []
    for x_batch, y in loader:
        logits, _ = model(x_batch)
        if num_classes <= 2:
            prob = torch.sigmoid(logits).view(-1, 1)  # [N,1]
            prob = torch.cat([1 - prob, prob], dim=1)  # [N,2]
        else:
            prob = F.softmax(logits, dim=1)            # [N,C]
        pred = prob.argmax(dim=1)

        ys.append(y.cpu().numpy())
        probs_list.append(prob.cpu().numpy())
        preds_list.append(pred.cpu().numpy())

    y_true = np.concatenate(ys, axis=0)
    y_prob = np.concatenate(probs_list, axis=0)
    y_pred = np.concatenate(preds_list, axis=0)

    # AUC
    try:
        if y_prob.shape[1] == 2:
            auc = roc_auc_score(y_true, y_prob[:, 1])
        else:
            auc = roc_auc_score(y_true, y_prob, multi_class="ovr")
    except Exception:
        auc = float("nan")

    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return {"auc": auc, "acc": acc, "f1_macro": f1m}


def train_one_epoch(model, loader, optimizer) -> float:
    model.train()
    loss_sum, count = 0.0, 0
    for x_batch, y in tqdm(loader, leave=False):
        logits, loss = model(x_batch, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * y.size(0)
        count += y.size(0)
    return loss_sum / max(count, 1)


def col_partitions_from_table(table, target_col: str) -> Tuple[List[str], List[str], List[str], int]:
    col_types = table.col_types
    cat_cols = [c for c, t in col_types.items() if t == ColType.CATEGORICAL and c != target_col]
    num_cols = [c for c, t in col_types.items() if t == ColType.NUMERICAL]
    bin_cols = [c for c, t in col_types.items() if t == ColType.BINARY]
    num_classes = table.num_classes
    return cat_cols, num_cols, bin_cols, num_classes


class EarlyStopperAUC:
    def __init__(self, patience: int = 5, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = float(min_delta)
        self.best = -np.inf
        self.counter = 0
        self.best_state = None

    def step(self, val_auc: float, model: torch.nn.Module) -> bool:
        if val_auc > self.best + self.min_delta:
            self.best = val_auc
            self.counter = 0
            self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience


def _split_one_type(feats: List[str], rng: np.random.RandomState) -> Tuple[List[str], List[str], List[str]]:
    feats = feats.copy()
    rng.shuffle(feats)
    n = len(feats)
    n_overlap = int(round(n / 3.0))
    rest = n - n_overlap
    n_aonly = int(round(rest / 2.0))
    # n_bonly = rest - n_aonly
    overlap = feats[:n_overlap]
    A = feats[n_overlap:n_overlap + n_aonly]
    B = feats[n_overlap + n_aonly:]
    return overlap, A, B


def split_columns_50_overlap(
    table,
    target_col: str,
    seed: int = 42,
) -> Tuple[List[str], List[str]]:
    rng = np.random.RandomState(seed)
    col_types = table.col_types

    cats = [c for c, t in col_types.items() if t == ColType.CATEGORICAL and c != target_col]
    nums = [c for c, t in col_types.items() if t == ColType.NUMERICAL]
    bins = [c for c, t in col_types.items() if t == ColType.BINARY]

    O_c, A_c, B_c = _split_one_type(cats, rng)
    O_n, A_n, B_n = _split_one_type(nums, rng)
    O_b, A_b, B_b = _split_one_type(bins, rng)

    overlap = O_c + O_n + O_b
    a_only = A_c + A_n + A_b
    b_only = B_c + B_n + B_b

    A_cols = overlap + a_only
    B_cols = overlap + b_only

    def _dedup(seq: List[str]) -> List[str]:
        seen, out = set(), []
        for x in seq:
            if x not in seen:
                out.append(x)
                seen.add(x)
        return out

    return _dedup(A_cols), _dedup(B_cols)


class SubTable:
    def __init__(self, base_table, keep_cols: List[str], target_col: str):
        self.base_table = base_table
        self.target_col = target_col

        cols = [c for c in keep_cols if c != target_col] + [target_col]
        self.df = base_table.df[cols].copy()

        self.col_types = {c: t for c, t in base_table.col_types.items() if c in self.df.columns}
        self.num_classes = base_table.num_classes

        self.train_mask = base_table.train_mask
        self.val_mask = getattr(base_table, "val_mask", None)
        self.test_mask = getattr(base_table, "test_mask", None)

        self.get_label_ids = base_table.get_label_ids


def ensure_masks(
    table,
    target_col: str,
    seed: int = 42,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    stratify: bool = True,
):
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
    if stratify:
        strat_temp = y_raw[temp_idx]
    else:
        strat_temp = None
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


if __name__ == "__main__":
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    td_base = Adult(cached_dir=args.cached_dir)[0]
    target = td_base.target_col
    ensure_masks(td_base, target_col=target, seed=args.seed, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2)

    A_cols, B_cols = split_columns_50_overlap(td_base, target_col=target, seed=args.seed)
    tdA = SubTable(td_base, keep_cols=A_cols, target_col=target)
    tdB = SubTable(td_base, keep_cols=B_cols, target_col=target)

    # Pretrain on subset A
    train_idxA = mask_to_index(tdA.train_mask)
    val_idxA = mask_to_index(tdA.val_mask)
    test_idxA = mask_to_index(tdA.test_mask) if getattr(tdA, "test_mask", None) is not None else None

    collateA = make_collate_td(tdA, target, device)
    train_loaderA = DataLoader(train_idxA.tolist(), batch_size=args.batch_size, shuffle=True, collate_fn=collateA)
    val_loaderA = DataLoader(val_idxA.tolist(), batch_size=args.batch_size, shuffle=False, collate_fn=collateA)
    test_loaderA = (DataLoader(test_idxA.tolist(), batch_size=args.batch_size, shuffle=False, collate_fn=collateA)
                    if test_idxA is not None else None)

    catA, numA, binA, nclsA = col_partitions_from_table(tdA, target)

    model = TransTabClassifier(
        categorical_columns=catA, numerical_columns=numA, binary_columns=binA,
        num_class=nclsA,
        hidden_dim=args.hidden_dim, num_layer=args.num_layers, num_attention_head=args.num_heads,
        hidden_dropout_prob=0.1, ffn_dim=args.hidden_dim * 2, activation="relu", device=device,
    ).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    es_pre = EarlyStopperAUC(patience=args.patience_pre)

    best_val_auc_pre = -np.inf
    test_at_best_pre = {"auc": float("-inf"), "acc": float("-inf"), "f1_macro": float("-inf")}

    for epoch in range(1, args.pre_epochs + 1):
        t0 = time.time()
        tr_loss = train_one_epoch(model, train_loaderA, optim)
        m_train = evaluate(model, train_loaderA, nclsA)
        m_val = evaluate(model, val_loaderA, nclsA)

        if m_val["auc"] > best_val_auc_pre:
            best_val_auc_pre = m_val["auc"]
            if test_loaderA is not None:
                m_testA = evaluate(model, test_loaderA, nclsA)
            else:
                m_testA = {"auc": np.nan, "acc": np.nan, "f1_macro": np.nan}
            test_at_best_pre = m_testA.copy()

        dt = time.time() - t0
        print(f"[Pre]  Epoch {epoch:02d}/{args.pre_epochs}  "
              f"Loss {tr_loss:.4f}  "
              f"Train AUC {m_train['auc']:.4f}  Val AUC {m_val['auc']:.4f}  "
              f"Val Acc {m_val['acc']:.4f}  Val F1 {m_val['f1_macro']:.4f}  ({dt:.2f}s)")

        if es_pre.step(m_val["auc"], model):
            print(f"[Pre]  Early stopping triggered (patience={args.patience_pre}).")
            break

    if es_pre.best_state is not None:
        model.load_state_dict(es_pre.best_state)
    print("\n[Pre ] Test @ Best Val on subset A:")
    print(f" AUC  {test_at_best_pre['auc']:.4f}")
    print(f" Acc  {test_at_best_pre['acc']:.4f}")
    print(f" F1m  {test_at_best_pre['f1_macro']:.4f}")

    # Fine-tune on subset B
    train_idxB = mask_to_index(tdB.train_mask)
    val_idxB = mask_to_index(tdB.val_mask)
    test_idxB = mask_to_index(tdB.test_mask) if getattr(tdB, "test_mask", None) is not None else None

    collateB = make_collate_td(tdB, target, device)
    train_loaderB = DataLoader(train_idxB.tolist(), batch_size=args.batch_size, shuffle=True, collate_fn=collateB)
    val_loaderB = DataLoader(val_idxB.tolist(), batch_size=args.batch_size, shuffle=False, collate_fn=collateB)
    test_loaderB = (DataLoader(test_idxB.tolist(), batch_size=args.batch_size, shuffle=False, collate_fn=collateB)
                    if test_idxB is not None else None)

    catB, numB, binB, nclsB = col_partitions_from_table(tdB, target)

    model.update({"cat": catB, "num": numB, "bin": binB, "num_class": nclsB})
    model.to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    es_ft = EarlyStopperAUC(patience=args.patience_ft)

    best_val_auc = -np.inf
    test_at_best = {"auc": float("-inf"), "acc": float("-inf"), "f1_macro": float("-inf")}

    for epoch in range(1, args.finetune_epochs + 1):
        t0 = time.time()
        tr_loss = train_one_epoch(model, train_loaderB, optim)
        m_train = evaluate(model, train_loaderB, nclsB)
        m_val = evaluate(model, val_loaderB, nclsB)
        if test_loaderB is None:
            m_test = {"auc": np.nan, "acc": np.nan, "f1_macro": np.nan}
        else:
            m_test = evaluate(model, test_loaderB, nclsB)
        dt = time.time() - t0

        if m_val["auc"] > best_val_auc:
            best_val_auc = m_val["auc"]
            test_at_best = m_test.copy()

        print(f"[FT ]  Epoch {epoch:02d}/{args.finetune_epochs}  "
              f"Loss {tr_loss:.4f}  "
              f"Train AUC {m_train['auc']:.4f}  Val AUC {m_val['auc']:.4f}  Test AUC {m_test['auc']:.4f}  "
              f"Val Acc {m_val['acc']:.4f}  Test Acc {m_test['acc']:.4f}  "
              f"Val F1 {m_val['f1_macro']:.4f}  Test F1 {m_test['f1_macro']:.4f}  ({dt:.2f}s)")

        if es_ft.step(m_val["auc"], model):
            print(f"[FT ]  Early stopping triggered (patience={args.patience_ft}).")
            break

    if es_ft.best_state is not None:
        model.load_state_dict(es_ft.best_state)

    final_test = (
        evaluate(model, test_loaderB, nclsB)
        if test_loaderB is not None
        else {"auc": np.nan, "acc": np.nan, "f1_macro": np.nan}
    )

    print("\n[Final] Test @ Best Val (tracked during training):")
    print(f" AUC  {test_at_best['auc']:.4f}")
    print(f" Acc  {test_at_best['acc']:.4f}")
    print(f" F1m  {test_at_best['f1_macro']:.4f}")

    print("\n[Final] Test after loading best FT weights:")
    print(f" AUC  {final_test['auc']:.4f}")
    print(f" Acc  {final_test['acc']:.4f}")
    print(f" F1m  {final_test['f1_macro']:.4f}")
