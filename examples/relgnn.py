# The RDL method from the "RelGNN: Composite Message Passing for Relational Deep Learning" paper.
# ArXiv: https://arxiv.org/abs/2502.06784
# Datasets          Rel-F1

# Tasks       driver-dnf   driver-top3   driver-position
# Metrics     ROC-AUC      ROC-AUC       MAE
# Rept.       75.29        85.69         3.798
# Ours        74.14        79.05         5.508
# Time(s)     100.67       27.25         86.13

import time
import argparse
import sys

import torch
import numpy as np
from sklearn.metrics import roc_auc_score, mean_absolute_error

sys.path.append("./")
sys.path.append("../")
sys.path.append("../../")
from rllm.datasets import RelF1Dataset
from rllm.dataloader import RelbenchLoader
from rllm.nn.models import RelGNNModel
from rllm.utils import get_atomic_routes


def train(model, optimizer, loss_fn, train_loader, target_table):
    model.train()
    total_loss = total_cnt = 0
    for batch in train_loader:
        batch.to(device)
        optimizer.zero_grad()
        out = model(batch, target_table)
        out = out.squeeze()
        y = batch[target_table].y
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y.size(0)
        total_cnt += y.size(0)
    return total_loss / total_cnt


def test(model, loader, target_table, clamp_min, clamp_max):
    model.eval()
    preds = []
    ys = []
    with torch.no_grad():
        for batch in loader:
            batch.to(device)
            pred = model(batch, target_table)
            if clamp_max is not None and clamp_min is not None:
                pred = torch.clamp(pred, clamp_min, clamp_max)
            else:
                pred = torch.sigmoid(pred)
            preds.append(pred.detach().cpu())
            ys.append(batch[target_table].y.detach().cpu())
    preds = torch.cat(preds, dim=0)
    ys = torch.cat(ys, dim=0)
    if clamp_max is not None and clamp_min is not None:
        metric = mean_absolute_error(ys.numpy(), preds.numpy())
    else:
        metric = roc_auc_score(ys.numpy(), preds.numpy())
    return metric


def main(args):
    # Data prepare
    dataset = RelF1Dataset(cached_dir=args.cache_dir)
    task = dataset.task_dict[args.task]
    target_table = task.entity_table
    col_stats_dict = dataset.tabledata_stats_dict
    (
        train_loader,
        val_loader,
        test_loader,
    ) = RelbenchLoader.get_loaders(
        dataset=dataset,
        task=args.task,
        batch_size=args.batch_size,
        num_neighbors=[args.num_neighbors // i for i in range(1, args.num_layers + 1)],
        to_bidirectional=True if args.task == "driver-position" else False,
    )
    # Model prepare

    atomic_routes_edge_types = get_atomic_routes(dataset.hdata.edge_types)

    if args.task == "driver-top3":
        model_config = {
            'relgnn_num_layers': 1,
            'hidden_dim': 128,
            'relgnn_aggr': 'sum',
            'relgnn_num_heads': 2,
        }
    elif args.task == "driver-dnf":
        model_config = {
            'relgnn_num_layers': 2,
            'hidden_dim': 128,
            'relgnn_aggr': 'sum',
            'relgnn_num_heads': 1,
        }
    elif args.task == "driver-position":
        model_config = {
            'relgnn_num_layers': 1,
            'hidden_dim': 128,
            'relgnn_aggr': 'sum',
            'relgnn_num_heads': 4,
        }
    model = RelGNNModel(
        data=dataset.hdata,
        col_stats_dict=col_stats_dict,
        atomic_routes_edge_types=atomic_routes_edge_types,
        out_dim=1,  # Binary classification
        use_temporal_encoder=True,
        **model_config,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.BCEWithLogitsLoss() \
        if args.task in ["driver-dnf", "driver-top3"] \
        else torch.nn.L1Loss()
    
    if args.task == "driver-position":
        clamp_min, clamp_max = np.percentile(
        task.task_data_dict['train'][0][task.target_col].to_numpy(), [2, 98]
    )
    else:
        clamp_min, clamp_max = None, None

    # Train and evaluate
    times = []
    metric = "ROC-AUC" if args.task in ["driver-dnf", "driver-top3"] else "MAE"
    best_val_metric = test_metric_at_best_val = 0
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss = train(model, optimizer, loss_fn, train_loader, target_table)
        train_metric = test(model, train_loader, target_table, clamp_min, clamp_max)
        val_metric = test(model, val_loader, target_table, clamp_min, clamp_max)
        test_metric = test(model, test_loader, target_table, clamp_min, clamp_max)

        if args.task in ["driver-dnf", "driver-top3"] and val_metric > best_val_metric:
            best_val_metric = val_metric
            test_metric_at_best_val = test_metric
        elif args.task == "driver-position" and val_metric < best_val_metric:
            best_val_metric = val_metric
            test_metric_at_best_val = test_metric

        times.append(time.time() - start)
        print(
            f"Epoch: [{epoch}/{args.epochs}]"
            f"Train Loss: {train_loss:.4f} Train {metric}: {train_metric:.4f} "
            f"Val {metric}: {val_metric:.4f}, Test {metric}: {test_metric:.4f} "
        )

    print(f"Mean time per epoch: {torch.tensor(times).mean():.4f}s")
    print(f"Total time: {sum(times):.4f}s")
    print(f"Test {metric} at best Val: {test_metric_at_best_val:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="driver-dnf")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_neighbors", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--cache_dir", type=str, default="../data/")

    args = parser.parse_args()

    assert args.task in ["driver-dnf", "driver-top3", "driver-position"], \
        "Only 'driver-dnf', 'driver-top3', and 'driver-position' tasks are supported."

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    main(args)
