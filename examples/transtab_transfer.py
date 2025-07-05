import argparse
import time
import os
import os.path as osp
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# Imports for your custom datasets and model
from rllm.datasets import MSTrafficSeattleDataset, MSTrafficMarylandDataset
from rllm.nn.models import TransTabClassifier

def parse_args():
    parser = argparse.ArgumentParser(description="TransTab Transfer Learning Test")
    parser.add_argument("--cached_dir", type=str, default="./data", help="Root directory for datasets")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--pre_epochs", type=int, default=15, help="Max epochs for pre-training")
    parser.add_argument("--finetune_epochs", type=int, default=25, help="Max epochs for fine-tuning")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--wd", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Transformer hidden dim")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of transformer layers")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--patience_pre", type=int, default=5, help="Early stopping patience for pre-training")
    parser.add_argument("--patience_ft", type=int, default=5, help="Early stopping patience for fine-tuning")
    return parser.parse_args()


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    Saves the best model to `path`.
    """
    def __init__(self, patience=3, verbose=False, delta=0.0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self._save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self._save_checkpoint(val_loss, model)
            self.counter = 0

    def _save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decreases."""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.4f} --> {val_loss:.4f}).  Saving model to {self.path}')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def validate(model, loader, device):
    """Compute average loss on validation set."""
    model.eval()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for batch_df, y in loader:
            y = y.to(device)
            # model(batch_df, y) 返回 (logits, loss)
            _, loss = model(batch_df, y)
            total_loss += loss.item() * y.size(0)
            total_count += y.size(0)
    return total_loss / total_count

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


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_count = 0
    for batch_df, y in loader:
        y = y.to(device)
        logits, loss = model(batch_df, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y.size(0)
        total_count += y.size(0)
    return total_loss / total_count


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_logits = []
    all_labels = []
    for batch_df, y in loader:
        y = y.to(device)
        out = model(batch_df)
        logits = out[0] if isinstance(out, tuple) else out
        all_logits.append(logits.cpu())
        all_labels.append(y.cpu())
    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0).numpy()
    probs = F.softmax(logits, dim=1).numpy()
    preds = np.argmax(probs, axis=1)
    return labels, probs, preds


if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------- Stage 1: Pre-training on Dataset1 ---------
    print("Loading Dataset1 for pre-training...")
    td1 = MSTrafficSeattleDataset(cached_dir=args.cached_dir)[0]
    df1 = td1.df.copy()
    target1 = td1.target_col

    # Encode labels
    le1 = LabelEncoder()
    df1[target1] = le1.fit_transform(df1[target1])

    # Split by mask
    train_df1 = df1.loc[td1.train_mask.numpy()]
    val_df1   = df1.loc[td1.val_mask.numpy()]

    # DataLoaders
    train_loader1 = DataLoader(
        DataFrameDataset(train_df1, target1),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader1 = DataLoader(
        DataFrameDataset(val_df1, target1),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # Column metadata
    col_types1 = td1.col_types
    cat_cols1 = [c for c, t in col_types1.items() if t.name == "CATEGORICAL" and c != target1]
    num_cols1 = [c for c, t in col_types1.items() if t.name == "NUMERICAL"]
    bin_cols1 = [c for c, t in col_types1.items() if t.name == "BINARY"]

    # Model initialization
    model = TransTabClassifier(
        categorical_columns=cat_cols1,
        numerical_columns=num_cols1,
        binary_columns=bin_cols1,
        num_class=td1.num_classes,
        hidden_dim=args.hidden_dim,
        num_layer=args.num_layers,
        num_attention_head=args.num_heads,
        hidden_dropout_prob=0.1,
        ffn_dim=args.hidden_dim * 2,
        activation="relu",
        device=device,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # EarlyStopping for pre-training
    pre_ckpt = "pretrain_checkpoint.pt"
    early_stopping = EarlyStopping(patience=args.patience_pre, verbose=True, path=pre_ckpt)

    print("Starting pre-training...")
    for epoch in range(1, args.pre_epochs + 1):
        start = time.time()
        train_loss = train_one_epoch(model, train_loader1, optimizer, device)
        val_loss = validate(model, val_loader1, device)
        print(f"[Pre-train] Epoch {epoch}/{args.pre_epochs}  Train Loss: {train_loss:.4f}  Val Loss: {val_loss:.4f}  Time: {time.time() - start:.2f}s")
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered in pre-training.")
            break

    # Load best pre-trained weights, then save in existing ckpt folder
    model.load_state_dict(torch.load(pre_ckpt))
    model.save("./ckpt_mstraffic/pretrained")
    model.load("./ckpt_mstraffic/pretrained")

    # --------- Stage 2: Fine-tuning on Dataset2 ---------
    print("Loading Dataset2 for fine-tuning...")
    td2 = MSTrafficMarylandDataset(cached_dir=args.cached_dir)[0]
    df2 = td2.df.copy()
    target2 = td2.target_col

    # Encode labels
    le2 = LabelEncoder()
    df2[target2] = le2.fit_transform(df2[target2])

    # Split by mask
    train_df2 = df2.loc[td2.train_mask.numpy()]
    val_df2   = df2.loc[td2.val_mask.numpy()]
    test_df2  = df2.loc[td2.test_mask.numpy()]

    # DataLoaders
    train_loader2 = DataLoader(
        DataFrameDataset(train_df2, target2),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader2 = DataLoader(
        DataFrameDataset(val_df2, target2),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    test_loader2 = DataLoader(
        DataFrameDataset(test_df2, target2),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # Column metadata for Dataset2
    col_types2 = td2.col_types
    cat_cols2 = [c for c, t in col_types2.items() if t.name == "CATEGORICAL" and c != target2]
    num_cols2 = [c for c, t in col_types2.items() if t.name == "NUMERICAL"]
    bin_cols2 = [c for c, t in col_types2.items() if t.name == "BINARY"]

    # Update model in-place for new dataset
    print("Updating model for Dataset2...")
    model.update({
        "cat":       cat_cols2,
        "num":       num_cols2,
        "bin":       bin_cols2,
        "num_class": td2.num_classes,
    })
    model.to(device)

    # Reinitialize optimizer for fine-tuning
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # EarlyStopping for fine-tuning
    ft_ckpt = "finetune_checkpoint.pt"
    early_stopping_ft = EarlyStopping(patience=args.patience_ft, verbose=True, path=ft_ckpt)

    print("Starting fine-tuning...")
    for epoch in range(1, args.finetune_epochs + 1):
        start = time.time()
        train_loss = train_one_epoch(model, train_loader2, optimizer, device)
        val_loss = validate(model, val_loader2, device)
        print(f"[Fine-tune] Epoch {epoch}/{args.finetune_epochs}  Train Loss: {train_loss:.4f}  Val Loss: {val_loss:.4f}  Time: {time.time() - start:.2f}s")
        early_stopping_ft(val_loss, model)
        if early_stopping_ft.early_stop:
            print("Early stopping triggered in fine-tuning.")
            break

    # Load best fine-tuned weights before final evaluation
    model.load_state_dict(torch.load(ft_ckpt))

    # --------- Evaluation ---------
    print("Evaluating on test set...")
    y_true, y_prob, y_pred = evaluate(model, test_loader2, device)

    auc = roc_auc_score(y_true, y_prob, multi_class="ovr")
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted")
    rec = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")

    print(f"AUC Score: {auc:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision (weighted): {prec:.4f}")
    print(f"Recall (weighted): {rec:.4f}")
    print(f"F1 Score (weighted): {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, digits=4))
