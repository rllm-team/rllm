# The RDL method from the "RelBench: A Benchmark for Deep Learning on Relational Databases" paper.
# ArXiv: https://arxiv.org/abs/2407.20060
# Datasets          Rel-F1

# Tasks       driver-dnf   driver-top3   driver-position
# Metrics     ROC-AUC      ROC-AUC       MAE
# Rept.       72.62        75.54         4.022
# Ours        74.14        75.79         4.001
# Time(s)     106.78       29.09         86.13

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
from rllm.nn.models import RDL


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
    model = RDL(
        data=dataset.hdata,
        col_stats_dict=col_stats_dict,
        hidden_dim=args.hidden_dim,
        out_dim=1,  # Regression
        hgnn_num_layers=args.num_layers,
        use_temporal_encoder=True,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.BCEWithLogitsLoss() \
        if args.task in ["driver-dnf", "driver-position"] \
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
    parser.add_argument("--hidden_dim", type=int, default=128)
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
