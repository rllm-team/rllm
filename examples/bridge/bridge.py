# The BRIDGE method from the "rLLM: Relational Table Learning with LLMs" paper.
# ArXiv: https://arxiv.org/abs/2407.20157

# Datasets      TLF2K   TML1M   TACM12K
# Acc(rept.)    0.422   0.362   0.256
# Acc(ours)     0.477   0.401   0.297
# Time(s)       32.48   372.33  28.61

import time
import argparse
import sys
import os.path as osp

import torch
import torch.nn.functional as F

sys.path.append("./")
sys.path.append("../")
sys.path.append("../../")
from rllm.datasets import TLF2KDataset, TACM12KDataset, TML1MDataset
from rllm.nn.conv.graph_conv import GCNConv
from rllm.nn.conv.table_conv import TabTransformerConv
from rllm.nn.models import BRIDGE, TableEncoder, GraphEncoder
from examples.bridge.utils import data_prepare


def build_bridge_model(num_classes, metadata, emb_size):
    t_encoder = TableEncoder(
        in_dim=emb_size,
        out_dim=emb_size,
        table_conv=TabTransformerConv,
        metadata=metadata,
    )
    g_encoder = GraphEncoder(
        in_dim=emb_size,
        out_dim=num_classes,
        graph_conv=GCNConv,
    )
    model = BRIDGE(
        table_encoder=t_encoder,
        graph_encoder=g_encoder,
    )
    return model


def train(model, optimizer, target_table, non_table_embeddings, adj, y, train_mask):
    model.train()
    optimizer.zero_grad()
    logits = model(table=target_table, non_table=non_table_embeddings, adj=adj)
    loss = F.cross_entropy(logits[train_mask].squeeze(), y[train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def test(model, target_table, non_table_embeddings, adj, y, masks):
    model.eval()
    logits = model(table=target_table, non_table=non_table_embeddings, adj=adj)
    preds = logits.argmax(dim=1)
    accs = []
    for mask in masks:
        correct = float(preds[mask].eq(y[mask]).sum().item())
        accs.append(correct / int(mask.sum()))
    return accs


def train_bridge_model(model, target_table, non_table_embeddings, adj, epochs, lr, wd):
    y = target_table.y
    train_mask, val_mask, test_mask = (
        target_table.train_mask,
        target_table.val_mask,
        target_table.test_mask,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    metric = "Acc"
    best_val_acc = test_acc = 0
    times = []
    for epoch in range(1, epochs + 1):
        start = time.time()
        train_loss = train(model, optimizer, target_table, non_table_embeddings, adj, y, train_mask)
        train_acc, val_acc, tmp_test_acc = test(model, target_table, non_table_embeddings, adj, y, [train_mask, val_mask, test_mask])

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc

        times.append(time.time() - start)
        print(
            f"Epoch: [{epoch}/{epochs}]"
            f"Train Loss: {train_loss:.4f} Train {metric}: {train_acc:.4f} "
            f"Val {metric}: {val_acc:.4f}, Test {metric}: {tmp_test_acc:.4f} "
        )

    print(f"Mean time per epoch: {torch.tensor(times).mean():.4f}s")
    print(f"Total time: {sum(times):.4f}s")
    print(f"Test {metric} at best Val: {test_acc:.4f}")

    return model, best_val_acc, test_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="tlf2k", choices=["tlf2k", "tml1m", "tacm12k"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--wd", type=float, default=1e-4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data")
    if args.dataset == "tlf2k":
        dataset = TLF2KDataset(cached_dir=path, force_reload=True)
    elif args.dataset == "tml1m":
        dataset = TML1MDataset(cached_dir=path, force_reload=True)
    elif args.dataset == "tacm12k":
        dataset = TACM12KDataset(cached_dir=path, force_reload=True)

    target_table, non_table_embeddings, adj, emb_size = data_prepare(dataset, args.dataset, device)
    model = build_bridge_model(target_table.num_classes, target_table.metadata, emb_size).to(device)
    train_bridge_model(model, target_table, non_table_embeddings, adj, args.epochs, args.lr, args.wd)