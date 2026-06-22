# The MetaRTL method from the "MetaRTL: Meta-path Attention Enhanced
# Relational Table Learning" paper, on the Rel-F1 dataset.

# Datasets              Rel-F1
# Tasks                 driver-dnf
# Metrics               ROC-AUC
# Rept.                 74.18
# Ours                  75.26
# Time per epoch(s)     3.14s

import os
import time
import copy
import math
import random
import argparse
import os.path as osp
import sys

import torch
from sklearn.metrics import roc_auc_score

sys.path.append("./")
sys.path.append("../")
sys.path.append("../../")
from rllm.datasets import RelF1Dataset
from rllm.dataloader import RelbenchLoader
from rllm.nn.encoder import MetaRTLEncoder
from rllm.nn.models import MetaPathFusion
from rllm.transforms.utils import MetaPathProp


def train_stage1(model, optimizer, loss_fn, loader, target_table, max_steps, device):
    r"""Stage-1 mini-batch training step for binary classification on Rel-F1."""
    model.train()
    total_loss = total_cnt = steps = 0
    for batch in loader:
        batch.to(device)
        optimizer.zero_grad()
        out = model(batch, target_table).squeeze()
        y = batch[target_table].y
        loss = loss_fn(out, y.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y.size(0)
        total_cnt += y.size(0)
        steps += 1
        if steps >= max_steps:
            break
    return total_loss / total_cnt


@torch.no_grad()
def test_stage1(model, loader, target_table, device):
    r"""Compute ROC-AUC on a Rel-F1 ``NeighborLoader`` split."""
    model.eval()
    preds, ys = [], []
    for batch in loader:
        batch.to(device)
        pred = torch.sigmoid(model(batch, target_table))
        preds.append(pred.detach().cpu())
        ys.append(batch[target_table].y.detach().cpu())
    preds = torch.cat(preds, dim=0).view(-1).numpy()
    ys = torch.cat(ys, dim=0).view(-1).numpy()
    return roc_auc_score(ys, preds)


@torch.no_grad()
def build_metapath_features(model, loaders, target_table, prop, device):
    r"""Encode each split with the frozen MetaRDL encoder and propagate
    meta-path features, caching ``(x, batch_idx, y)`` triples per split."""
    model.eval()
    x_split = {}
    for split, loader in loaders.items():
        x_split[split] = []
        for batch in loader:
            batch.to(device)
            x_dict = model.encode(batch)
            x = prop(batch, x_dict)  # (B, M, D)
            x = x[: batch[target_table].seed_time.size(0)]
            y = batch[target_table].y.float()
            batch_idx = batch[target_table].input_id
            x_split[split].append((x.cpu(), batch_idx.cpu(), y.cpu()))
    return x_split


def train_fusion(fusion_model, optimizer, loss_fn, x_split, device):
    r"""Stage-2 fusion-model training over the cached meta-path tensors."""
    fusion_model.train()
    total_loss = total_cnt = 0
    random.shuffle(x_split["train"])
    for x, batch_idx, y in x_split["train"]:
        x, batch_idx, y = x.to(device), batch_idx.to(device), y.to(device)
        optimizer.zero_grad()
        pred = fusion_model(x, batch_idx).view(-1)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * pred.size(0)
        total_cnt += pred.size(0)
    return total_loss / total_cnt


@torch.no_grad()
def test_fusion(fusion_model, x_split, split, device):
    r"""ROC-AUC of the stage-2 fusion model on a cached split."""
    fusion_model.eval()
    preds, ys = [], []
    for x, batch_idx, y in x_split[split]:
        x, batch_idx = x.to(device), batch_idx.to(device)
        pred = torch.sigmoid(fusion_model(x, batch_idx))
        preds.append(pred.detach().cpu())
        ys.append(y)
    preds = torch.cat(preds, dim=0).view(-1).numpy()
    ys = torch.cat(ys, dim=0).view(-1).numpy()
    return roc_auc_score(ys, preds)


def main(args):
    # Data prepare
    dataset = RelF1Dataset(cached_dir=args.cache_dir)
    task = dataset.task_dict["driver-dnf"]
    target_table = task.entity_table
    col_stats_dict = dataset.tabledata_stats_dict

    train_loader, val_loader, test_loader = RelbenchLoader.get_loaders(
        dataset=dataset,
        task="driver-dnf",
        batch_size=args.batch_size,
        num_neighbors=[args.num_neighbors // i for i in range(1, args.num_layers + 1)],
        to_bidirectional=False,
    )
    loaders = {"train": train_loader, "val": val_loader, "test": test_loader}

    loss_fn = torch.nn.BCEWithLogitsLoss()
    metric = "ROC-AUC"
    num_train_nodes = len(task.task_data_dict["train"][0])

    # encoder
    model = MetaRTLEncoder(
        data=dataset.hdata,
        col_stats_dict=col_stats_dict,
        hidden_dim=args.hidden_dim,
        out_dim=1,
        gnn=args.gnn,
        gnn_num_layers=args.num_layers,
        use_temporal_encoder=True,
        reg_task=False,
    ).to(device)

    encoder_path = osp.join(args.cache_dir, f"metartl_driver-dnf_{args.gnn}_encoder.pt")
    if args.skip_stage1 and osp.exists(encoder_path):
        model.load_state_dict(torch.load(encoder_path, map_location=device))
        print(f"====> Loaded stage1 encoder from {encoder_path}")
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        best_val = -math.inf
        best_state = copy.deepcopy(model.state_dict())
        for epoch in range(1, args.stage1_epochs + 1):
            loss = train_stage1(
                model, optimizer, loss_fn, train_loader,
                target_table, args.max_steps_per_epoch, device,
            )
            val_metric = test_stage1(model, val_loader, target_table, device)
            if val_metric > best_val:
                best_val = val_metric
                best_state = copy.deepcopy(model.state_dict())
            print(f"[Stage1] Epoch {epoch:02d} Loss {loss:.4f} Val {metric} {val_metric:.4f}")
        model.load_state_dict(best_state)
        torch.save(model.state_dict(), encoder_path)
        print(f"====> Saved stage1 encoder to {encoder_path}")

    # meta-path propagation
    prop = MetaPathProp(
        target_node_type=target_table,
        min_hops=0,
        max_hops=args.max_hops,
        edge_schema=dataset.hdata.edge_types,
    )
    x_split = build_metapath_features(model, loaders, target_table, prop, device)
    print("====> Meta-path features prepared")

    in_dim = x_split["train"][0][0].size(-1)
    num_metapaths = x_split["train"][0][0].size(1)

    # fusion
    fusion_model = MetaPathFusion(
        num_nodes=num_train_nodes,
        in_dim=in_dim,
        hidden_dim=in_dim,
        num_metapaths=num_metapaths,
        out_dim=1,
        num_centroids=args.num_centroids,
        num_heads=args.num_heads,
        att_drop=0.3,
        readout_drop=0.3,
        readout_layers=1,
        norm="batch_norm",
    ).to(device)
    optimizer2 = torch.optim.Adam(fusion_model.parameters(), lr=args.lr)

    best_val = -math.inf
    test_at_best_val = best_val
    times = []
    for epoch in range(1, args.stage2_epochs + 1):
        start = time.time()
        loss = train_fusion(fusion_model, optimizer2, loss_fn, x_split, device)
        val_metric = test_fusion(fusion_model, x_split, "val", device)
        test_metric = test_fusion(fusion_model, x_split, "test", device)
        if val_metric > best_val:
            best_val = val_metric
            test_at_best_val = test_metric
        times.append(time.time() - start)
        print(
            f"[Stage2] Epoch {epoch:02d} Loss {loss:.4f} "
            f"Val {metric} {val_metric:.4f} Test {metric} {test_metric:.4f}"
        )

    print(f"Mean time per stage2 epoch: {torch.tensor(times).mean():.4f}s")
    print(f"Test {metric} at best Val: {test_at_best_val:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gnn", type=str, default="SAGE", choices=["SAGE", "RelGNN"])
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_neighbors", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--stage1_epochs", type=int, default=5)
    parser.add_argument("--stage2_epochs", type=int, default=50)
    parser.add_argument("--max_steps_per_epoch", type=int, default=2000)
    parser.add_argument("--max_hops", type=int, default=3)
    parser.add_argument("--num_centroids", type=int, default=4096)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--skip_stage1", action="store_true")
    parser.add_argument("--cache_dir", type=str, default="./data/")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    main(args)
