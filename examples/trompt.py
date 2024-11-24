# The Trompt method from the
# "Trompt: Towards a Better Deep Neural Network for Tabular Data" paper.
# ArXiv: https://arxiv.org/abs/1609.02907

# Datasets  Titanic    Adult
# Acc       0.809      0.862
# Time      13.9s      911.7s

import argparse
import sys
import time
from typing import Any, Dict, List
import os.path as osp

from tqdm import tqdm
import torch
from torch import Tensor
from torch.utils.data import DataLoader
import torch.nn.functional as F

sys.path.append("./")
sys.path.append("../")
from rllm.types import ColType
from rllm.datasets.adult import Adult
from rllm.transforms.table_transforms import TromptTransform
from rllm.nn.conv.table_conv import TromptConv

parser = argparse.ArgumentParser()
parser.add_argument("--dim", help="embedding dim", type=int, default=128)
parser.add_argument("--num_layers", type=int, default=6)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--wd", type=float, default=5e-4)
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Prepare datasets
path = osp.join(osp.dirname(osp.realpath(__file__)), "..", "data")
dataset = Adult(cached_dir=path)[0]
dataset.to(device)

# Split dataset, here the ratio of train-val-test is 80%-10%-10%
train_loader, val_loader, test_loader = dataset.get_dataloader(
    26048, 6513, 16281, batch_size=args.batch_size
)


# Set up model and optimizer
class Trompt(torch.nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int,
        num_prompts: int,
        col_stats_dict: Dict[ColType, List[Dict[str, Any]]],
    ):
        super().__init__()
        self.out_dim = out_dim
        self.x_prompt = torch.nn.Parameter(torch.empty(num_prompts, hidden_dim))

        self.transforms = torch.nn.ModuleList(
            [
                TromptTransform(
                    out_dim=hidden_dim,
                    col_stats_dict=col_stats_dict,
                )
                for _ in range(num_layers)
            ]
        )
        self.convs = torch.nn.ModuleList(
            [
                TromptConv(
                    in_dim=in_dim,
                    hidden_dim=hidden_dim,
                    num_prompts=num_prompts,
                )
                for _ in range(num_layers)
            ]
        )

        self.linear = torch.nn.Linear(hidden_dim, 1)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.Linear(hidden_dim, out_dim),
        )
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.x_prompt)
        for conv in self.convs:
            conv.reset_parameters()
        torch.nn.init.xavier_uniform_(self.linear.weight)
        torch.nn.init.zeros_(self.linear.bias)
        for layer in self.mlp:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.zeros_(layer.bias)

    def forward(self, x) -> Tensor:
        outs = []
        batch_size = x[list(x.keys())[0]].size(0)
        x_prompt = self.x_prompt.unsqueeze(0).repeat(batch_size, 1, 1)
        for transform, conv in zip(self.transforms, self.convs):
            x_transform = transform(x)
            x_prompt = conv(x_transform, x_prompt)
            w_prompt = F.softmax(self.linear(x_prompt), dim=1)
            out = (w_prompt * x_prompt).sum(dim=1)
            out = self.mlp(out)
            out = out.reshape(batch_size, 1, self.out_dim)
            outs.append(out)
        return torch.cat(outs, dim=1).mean(dim=1)


model = Trompt(
    in_dim=dataset.num_cols,
    hidden_dim=args.dim,
    out_dim=dataset.num_classes,
    num_layers=args.num_layers,
    num_prompts=128,
    col_stats_dict=dataset.stats_dict,
).to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=args.lr,
    weight_decay=args.wd,
)


def train(epoch: int) -> float:
    model.train()
    loss_accum = total_count = 0
    for batch in tqdm(train_loader, desc=f"Epoch: {epoch}"):
        x, y = batch
        pred = model.forward(x)
        loss = F.cross_entropy(pred, y.long())
        optimizer.zero_grad()
        loss.backward()
        loss_accum += float(loss) * y.size(0)
        total_count += y.size(0)
        optimizer.step()
    return loss_accum / total_count


@torch.no_grad()
def test(loader: DataLoader) -> float:
    model.eval()
    correct = total = 0
    for batch in loader:
        feat_dict, y = batch
        pred = model.forward(feat_dict)
        _, predicted = torch.max(pred, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()
    accuracy = correct / total
    return accuracy


metric = "Acc"
best_val_metric = 0
best_test_metric = 0
times = []
st = time.time()
for epoch in range(1, args.epochs + 1):
    start = time.time()
    train_loss = train(epoch)
    train_metric = test(train_loader)
    val_metric = test(val_loader)
    test_metric = test(test_loader)

    if val_metric > best_val_metric:
        best_val_metric = val_metric
        best_test_metric = test_metric

    times.append(time.time() - start)
    print(
        f"Train Loss: {train_loss:.4f}, Train {metric}: {train_metric:.4f}, "
        f"Val {metric}: {val_metric:.4f}, Test {metric}: {test_metric:.4f}"
    )
    optimizer.step()
et = time.time()
print(f"Mean time per epoch: {torch.tensor(times).mean():.4f}s")
print(f"Total time: {et-st}s")
print(
    f"Best Val {metric}: {best_val_metric:.4f}, "
    f"Best Test {metric}: {best_test_metric:.4f}"
)
