import argparse
import os
import time

import torch
import pandas as pd
import numpy as np
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

from rllm.datasets import MSTrafficSeattleDataset, MSTrafficMarylandDataset
from rllm.nn.models import (
    TransTabForCL,
    TransTabClassifier,
)


def parse_args():
    parser = argparse.ArgumentParser("TransTab VPCL + Transfer Learning")
    parser.add_argument("--cached_dir",     type=str,   default="./data")
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--pre_epochs",     type=int,   default=20)
    parser.add_argument("--finetune_epochs",type=int,   default=20)
    parser.add_argument("--batch_size",     type=int,   default=64)
    parser.add_argument("--lr_pre",         type=float, default=1e-4)
    parser.add_argument("--lr_finetune",    type=float, default=1e-3)
    parser.add_argument("--wd",             type=float, default=1e-4)
    parser.add_argument("--hidden_dim",     type=int,   default=128)
    parser.add_argument("--num_layers",     type=int,   default=2)
    parser.add_argument("--num_heads",      type=int,   default=8)
    parser.add_argument("--num_partition",  type=int,   default=4)
    parser.add_argument("--overlap_ratio",  type=float, default=0.5)
    return parser.parse_args()


class CLDataset(Dataset):
    """Only returns DataFrame, for self-supervised contrastive learning"""
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return row.to_frame().T

def cl_collate_fn(batch):
    # Concat a batch of "single row DataFrames" into one large DataFrame
    return pd.concat(batch, ignore_index=True)

def train_one_epoch_cl(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_count = 0
    for batch_df in loader:
        _, loss = model(batch_df)           # forward returns (None, loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bs = len(batch_df)
        total_loss += loss.item() * bs
        total_count += bs
    return total_loss / total_count

def train_one_epoch_cls(model, loader, optimizer, device):
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
def evaluate_cls(model, loader, device):
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

class ClassificationDataset(Dataset):
    """Supervised classification dataset for fine-tuning"""
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

def cls_collate_fn(batch):
    x_list, y_list = zip(*batch)
    batch_df = pd.concat(x_list, ignore_index=True)
    y = torch.tensor(y_list, dtype=torch.long)
    return batch_df, y

if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    print("Loading Dataset1 (Seattle) for VPCL pretraining...")
    td1   = MSTrafficSeattleDataset(cached_dir=args.cached_dir)[0]
    df1   = td1.df.copy()

    train_df1 = df1.loc[td1.train_mask.numpy()]
    val_df1   = df1.loc[td1.val_mask.numpy()]

    # DataLoaders
    train_loader1 = DataLoader(
        CLDataset(train_df1),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=cl_collate_fn,
    )
    val_loader1 = DataLoader(
        CLDataset(val_df1),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=cl_collate_fn,
    )

    col_types1 = td1.col_types
    cat_cols1 = [c for c,t in col_types1.items() if t.name=="CATEGORICAL"]
    num_cols1 = [c for c,t in col_types1.items() if t.name=="NUMERICAL"]
    bin_cols1 = [c for c,t in col_types1.items() if t.name=="BINARY"]

    model_cl = TransTabForCL(
        categorical_columns=cat_cols1,
        numerical_columns=num_cols1,
        binary_columns=bin_cols1,
        supervised=False,
        num_partition=args.num_partition,
        overlap_ratio=args.overlap_ratio,
        hidden_dim=args.hidden_dim,
        num_layer=args.num_layers,
        num_attention_head=args.num_heads,
        hidden_dropout_prob=0.1,
        ffn_dim=args.hidden_dim*2,
        activation="relu",
        device=device,
    ).to(device)

    optimizer_cl = torch.optim.AdamW(
        model_cl.parameters(),
        lr=args.lr_pre,
        weight_decay=args.wd,
    )

    print("Starting VPCL pretraining...")
    for epoch in range(1, args.pre_epochs+1):
        start = time.time()
        loss = train_one_epoch_cl(model_cl, train_loader1, optimizer_cl, device)
        print(f"[VPCL] Epoch {epoch}/{args.pre_epochs}, Loss: {loss:.4f}, Time: {time.time()-start:.1f}s")

    ckpt_cl = "./ckpt_cl/pretrained"
    model_cl.save(ckpt_cl)
    print(f"Saved VPCL pretrained model to {ckpt_cl}\n")

    print("Loading Dataset2 (Maryland) for fine-tuning...")
    td2   = MSTrafficMarylandDataset(cached_dir=args.cached_dir)[0]
    df2   = td2.df.copy()
    target2 = td2.target_col

    le2 = LabelEncoder()
    df2[target2] = le2.fit_transform(df2[target2])

    train_df2 = df2.loc[td2.train_mask.numpy()]
    val_df2   = df2.loc[td2.val_mask.numpy()]
    test_df2  = df2.loc[td2.test_mask.numpy()]

    train_loader2 = DataLoader(
        ClassificationDataset(train_df2, target2),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=cls_collate_fn,
    )
    val_loader2 = DataLoader(
        ClassificationDataset(val_df2, target2),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=cls_collate_fn,
    )
    test_loader2 = DataLoader(
        ClassificationDataset(test_df2, target2),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=cls_collate_fn,
    )

    col_types2 = td2.col_types
    cat_cols2 = [c for c,t in col_types2.items() if t.name=="CATEGORICAL" and c!=target2]
    num_cols2 = [c for c,t in col_types2.items() if t.name=="NUMERICAL"]
    bin_cols2 = [c for c,t in col_types2.items() if t.name=="BINARY"]


    model_ds = TransTabClassifier(
        categorical_columns=cat_cols2,
        numerical_columns=num_cols2,
        binary_columns=bin_cols2,
        num_class=td2.num_classes,
        hidden_dim=args.hidden_dim,
        num_layer=args.num_layers,
        num_attention_head=args.num_heads,
        hidden_dropout_prob=0.1,
        ffn_dim=args.hidden_dim*2,
        activation="relu",
        device=device,
    ).to(device)

    # Load VPCL pre-trained weights (including embedding + transformer)
    model_ds.load(ckpt_cl)

    # Switch to Maryland columns & rebuild the category header
    model_ds.update({
        "cat":       cat_cols2,
        "num":       num_cols2,
        "bin":       bin_cols2,
        "num_class": td2.num_classes,
    })

    optimizer_ds = torch.optim.AdamW(
        model_ds.parameters(),
        lr=args.lr_finetune,
        weight_decay=args.wd,
    )

    print("Starting downstream fine-tuning...")
    for epoch in range(1, args.finetune_epochs+1):
        start = time.time()
        loss = train_one_epoch_cls(model_ds, train_loader2, optimizer_ds, device)
        print(f"[Fine-tune] Epoch {epoch}/{args.finetune_epochs}, Loss: {loss:.4f}, Time: {time.time()-start:.1f}s")

    print("\nEvaluating on test set...")
    y_true, y_prob, y_pred = evaluate_cls(model_ds, test_loader2, device)

    print(f"AUC Score: {roc_auc_score(y_true, y_prob, multi_class='ovr'):.4f}")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"F1 Score:  {f1_score(y_true, y_pred, average='weighted'):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, digits=4))
