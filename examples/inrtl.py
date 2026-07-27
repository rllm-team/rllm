# Datasets  TML1M       TLF2K       TACM12K
# Metrics   Acc         Acc         Acc
# Rept.     40.60       50.40       48.40
# Ours      40.40       45.80       47.80
# Time      50.89s      19.758s     18.41s

from __future__ import annotations

import argparse
import os.path as osp
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from rllm.datasets import TACM12KDataset, TLF2KDataset, TML1MDataset
from rllm.nn.conv.table_conv import TabTransformerConv
from rllm.nn.models import InRTL

from examples.bridge.utils import data_prepare


def build_model(target_table, emb_size, args):
    if args.dataset == "tacm12k":
        return InRTL(
            in_channels=emb_size,
            hidden_channels=args.hidden_dim,
            out_channels=target_table.num_classes,
            attn_num_layers=args.attn_num_layers,
            attn_num_heads=args.attn_num_heads,
            attn_dropout=args.attn_dropout,
            gnn_num_layers=args.gnn_num_layers,
            gnn_dropout=args.gnn_dropout,
            alpha=args.alpha,
            beta=args.beta,
            aggregate=args.aggregate,
            use_orig_x=args.use_orig_x,
        )

    return InRTL(
        in_channels=emb_size,
        hidden_channels=args.hidden_dim,
        out_channels=target_table.num_classes,
        table_metadata=target_table.metadata,
        attn_num_layers=args.attn_num_layers,
        attn_num_heads=args.attn_num_heads,
        attn_dropout=args.attn_dropout,
        gnn_num_layers=args.gnn_num_layers,
        gnn_dropout=args.gnn_dropout,
        alpha=args.alpha,
        beta=args.beta,
        aggregate=args.aggregate,
        use_orig_x=args.use_orig_x,
    )


def train_epoch(model, optimizer, target_table, non_table_embeddings, adj):
    model.train()
    optimizer.zero_grad()
    if not model.uses_table_encoder:
        logits = model(non_table_embeddings, adj)
    else:
        logits = model(target_table, non_table_embeddings, adj)
    loss = F.cross_entropy(
        logits[target_table.train_mask],
        target_table.y[target_table.train_mask],
    )
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, target_table, non_table_embeddings, adj):
    model.eval()
    if not model.uses_table_encoder:
        logits = model(non_table_embeddings, adj)
    else:
        logits = model(target_table, non_table_embeddings, adj)
    preds = logits.argmax(dim=1)
    accs = []
    for mask in [
        target_table.train_mask,
        target_table.val_mask,
        target_table.test_mask,
    ]:
        correct = preds[mask].eq(target_table.y[mask]).sum().item()
        accs.append(correct / int(mask.sum()))
    return accs


def load_dataset(dataset_name: str, cached_dir: str, force_reload: bool):
    if dataset_name == "tlf2k":
        return TLF2KDataset(cached_dir=cached_dir, force_reload=force_reload)
    if dataset_name == "tml1m":
        return TML1MDataset(cached_dir=cached_dir, force_reload=force_reload)
    if dataset_name == "tacm12k":
        return TACM12KDataset(cached_dir=cached_dir, force_reload=force_reload)
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cached_dir = osp.join(osp.dirname(osp.realpath(__file__)), "..", "data")

    dataset = load_dataset(args.dataset, cached_dir, args.force_reload)
    use_paper_embeddings = args.dataset == "tacm12k"
    target_table, non_table_embeddings, adj, emb_size = data_prepare(
        dataset,
        args.dataset,
        device,
        use_paper_embeddings=use_paper_embeddings,
    )

    model = build_model(target_table, emb_size, args).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.wd,
    )

    best_val_acc = 0.0
    best_test_acc = 0.0
    times = []
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss = train_epoch(
            model, optimizer, target_table, non_table_embeddings, adj
        )
        train_acc, val_acc, test_acc = evaluate(
            model, target_table, non_table_embeddings, adj
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
        times.append(time.time() - start)
        print(
            f"Epoch: [{epoch}/{args.epochs}] "
            f"Loss: {train_loss:.4f} "
            f"Train Acc: {train_acc:.4f} "
            f"Val Acc: {val_acc:.4f} "
            f"Test Acc: {test_acc:.4f}"
        )

    print(f"Mean time per epoch: {torch.tensor(times).mean():.4f}s")
    print(f"Best Val acc: {best_val_acc:.4f}")
    print(f"Best Test acc at best Val: {best_test_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="tml1m",
        choices=["tlf2k", "tml1m", "tacm12k"],
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=384)
    parser.add_argument("--attn_num_layers", type=int, default=1)
    parser.add_argument("--attn_num_heads", type=int, default=4)
    parser.add_argument("--attn_dropout", type=float, default=0.5)
    parser.add_argument("--gnn_num_layers", type=int, default=2)
    parser.add_argument("--gnn_dropout", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument(
        "--aggregate", type=str, default="add", choices=["add", "concat"]
    )
    parser.add_argument("--use_orig_x", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force_reload", action="store_true")
    args = parser.parse_args()

    main(args)
