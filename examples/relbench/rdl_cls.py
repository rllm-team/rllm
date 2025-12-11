# The RDL method from the "RelBench: A Benchmark for Deep Learning on Relational Databases" paper.
# ArXiv: https://arxiv.org/abs/2407.20060
# Datasets          Rel-F1
# TaskType          Binary Classification

# Tasks             driver-dnf   driver-top3
# ROC-AUC(rept.)    72.62        75.54
# ROC-AUC(ours)     74.14        75.79
# Time(s)          106.78        29.09

import time
import argparse
import sys

import torch
from sklearn.metrics import roc_auc_score

sys.path.append("./")
sys.path.append("../")
sys.path.append("../../")
from rllm.datasets import RelF1Dataset
from rllm.dataloader import RelbenchLoader
from rllm.nn.models.rdl import RDLModel


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


def test(model, loader, target_table):
    model.eval()
    preds = []
    ys = []
    with torch.no_grad():
        for batch in loader:
            batch.to(device)
            pred = model(batch, target_table)
            pred = torch.sigmoid(pred)
            preds.append(pred.detach().cpu())
            ys.append(batch[target_table].y.detach().cpu())
    preds = torch.cat(preds, dim=0)
    ys = torch.cat(ys, dim=0)
    auc = roc_auc_score(ys.numpy(), preds.numpy())
    return auc


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
    )
    # Model prepare
    model = RDLModel(
        data=dataset.hdata,
        col_stats_dict=col_stats_dict,
        hidden_dim=args.hidden_dim,
        out_dim=1,  # Binary classification
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    # Train and evaluate
    times = []
    best_val_auc = test_auc_at_best_val = 0
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss = train(model, optimizer, loss_fn, train_loader, target_table)
        train_auc = test(model, train_loader, target_table)
        val_auc = test(model, val_loader, target_table)
        test_auc = test(model, test_loader, target_table)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            test_auc_at_best_val = test_auc

        times.append(time.time() - start)
        print(
            f"Epoch: [{epoch}/{args.epochs}]"
            f"Train Loss: {train_loss:.4f} Train ROC-AUC: {train_auc:.4f} "
            f"Val ROC-AUC: {val_auc:.4f}, Test ROC-AUC: {test_auc:.4f} "
        )

    print(f"Mean time per epoch: {torch.tensor(times).mean():.4f}s")
    print(f"Total time: {sum(times):.4f}s")
    print(f"Test ROC-AUC at best Val: {test_auc_at_best_val:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="driver-dnf")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_neighbors", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--cache_dir", type=str, default="./data/")

    args = parser.parse_args()

    assert args.task in ["driver-dnf", "driver-top3"], \
        "Only 'driver-dnf' and 'driver-top3' tasks are supported."

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    main(args)
