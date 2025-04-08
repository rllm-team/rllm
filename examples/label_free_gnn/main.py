# Label-Free-GNN means to adopt LLMs as predictor to label some nodes, and then
# adopt GNNs for node classification. In the node selection phase, we compare
# the classical Random and VertexCover methods with PS-FeatProp-W (from "Label-free
# Node Classification on Graphs with Large Language Models (LLMS)" paper).
# In classification phase, we adopt the well-known GCN model.
# The results are:
# Datasets                   CiteSeer    Cora        PubMed
# Random                     0.6919      0.7162      0.7609
# VertexCover                0.7129      0.7072      0.7900
# PS-FeatProp-W              0.6915      0.7653      0.7731
# PS-FeatProp-W (Reported)   0.6864      0.7623      0.7884


import argparse
import os.path as osp
import sys
import time

import torch
import torch.nn.functional as F
from annotation.annotation import annotate
from langchain_community.llms import LlamaCpp

sys.path.append("../")

from node_selection.node_selection import active_generate_mask, post_filter
from rllm.datasets.tagdataset import TAGDataset
from rllm.transforms.graph_transforms import GCNTransform
from rllm.llm.llm_module.langchain_llm import LangChainLLM
from rllm.nn.conv.graph_conv import GCNConv

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    type=str,
    default="cora",
    choices=["cora", "citeseer", "pubmed"],
    help="dataset",
)
parser.add_argument(
    "--hidden_dim",
    type=int,
    default=64,
    help="number of hidden dim in GCN",
)
parser.add_argument(
    "--active_method",
    type=str,
    default="Random",
    choices=["Random", "VertexCover", "FeatProp"],
    help="active node selection",
)
parser.add_argument(
    "--post_filter",
    type=bool,
    default=False,
    help="perform post filtering",
)
parser.add_argument(
    "--filter_strategy",
    type=str,
    default="conf+density",
    choices=["conf+density", "conf+density+entropy"],
    help="strategy for post filtering",
)
parser.add_argument(
    "--weighted_loss",
    type=bool,
    default=False,
    help="use weighted loss",
)
parser.add_argument(
    "--n_tries",
    type=int,
    default=3,
    help="number of tries asking LLM",
)
parser.add_argument(
    "--budget",
    type=int,
    default=20,
    help="number of LLM queries per class",
)
parser.add_argument(
    "--lr",
    type=float,
    default=0.1,
    help="learning rate",
)
parser.add_argument("--wd", type=float, default=5e-4, help="weight decay")
parser.add_argument("--epochs", type=int, default=150, help="number of training epochs")
parser.add_argument("--use_cache", type=bool, default=True, help="use cache")
parser.add_argument("--val", type=bool, default=False, help="use validation set")
parser.add_argument("--n_rounds", type=int, default=20, help="number of rounds")
args = parser.parse_args()


path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data")

transform = GCNTransform(normalize_features="l2")
dataset = TAGDataset(
    path,
    args.dataset,
    use_cache=args.use_cache,
    transform=transform,
    force_reload=True,
)
data = dataset[0]

method = args.active_method
if args.post_filter:
    method = "PS-" + method
if args.weighted_loss:
    method += "-W"

train_mask, val_mask, test_mask = active_generate_mask(
    data, method=args.active_method, val=args.val, budget=args.budget
)

if not args.use_cache:
    model_path = "/path/to/llm"
    llm = LangChainLLM(LlamaCpp(model_path=model_path, n_gpu_layers=33))
    pl_indices = torch.nonzero(train_mask | val_mask, as_tuple=False).squeeze()
    data = annotate(data, pl_indices, llm, args.n_tries)

if args.post_filter:
    filtered_mask = post_filter(data, train_mask | val_mask, args.filter_strategy)
    train_mask = train_mask & filtered_mask
    val_mask = val_mask & filtered_mask


class GCN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)

    def forward(self, x, adj):
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.conv1(x, adj))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, adj)
        return x


class Trainer:
    def __init__(self, data, model, optimizer, masks, val, weighted_loss):
        self.data = data
        self.model = model
        self.optimizer = optimizer
        self.train_mask = masks["train_mask"]
        self.val_mask = masks["val_mask"]
        self.test_mask = masks["test_mask"]
        self.val = val
        self.weighted_loss = weighted_loss

    def train(self):
        self.model.train()
        self.optimizer.zero_grad()
        out = self.model(self.data.x, data.adj)
        loss_fn = torch.nn.CrossEntropyLoss()
        if self.weighted_loss:
            loss = (
                loss_fn(out[self.train_mask], self.data.pl[self.train_mask])
                * self.data.conf[self.train_mask].mean()
            )
        else:
            loss = loss_fn(out[train_mask], data.pl[train_mask])
        loss.backward()
        self.optimizer.step()
        return loss.item()

    @torch.no_grad()
    def test(self):
        self.model.eval()
        out = self.model(data.x, data.adj)
        pred = out.argmax(dim=1)

        accs = []
        correct = float(pred[train_mask].eq(data.pl[train_mask]).sum().item())
        accs.append(correct / int(train_mask.sum()))

        if self.val:
            correct = float(pred[val_mask].eq(data.pl[val_mask]).sum().item())
            accs.append(correct / int(val_mask.sum()))

        correct = float(pred[test_mask].eq(data.y[test_mask]).sum().item())
        accs.append(correct / int(test_mask.sum()))

        return accs


acc_list = []


model = GCN(
    in_dim=data.x.shape[1],
    hidden_dim=args.hidden_dim,
    out_dim=data.num_classes,
)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
masks = {"train_mask": train_mask, "val_mask": val_mask, "test_mask": test_mask}

trainer = Trainer(data, model, optimizer, masks, args.val, args.weighted_loss)

metric = "Acc"
best_val_acc = best_test_acc = 0
times = []
for epoch in range(1, args.epochs + 1):
    start = time.time()

    train_loss = trainer.train()

    if args.val:
        train_acc, val_acc, test_acc = trainer.test()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
    else:
        train_acc, test_acc = trainer.test()
        if test_acc > best_test_acc:
            best_test_acc = test_acc

    times.append(time.time() - start)

    if args.val:
        print(
            f"Epoch: [{epoch}/{args.epochs}] "
            f"Train Loss: {train_loss:.4f} Train {metric}: {train_acc:.4f} "
            f"Val {metric}: {val_acc:.4f}, Test {metric}: {test_acc:.4f} "
        )
    else:
        print(
            f"Epoch: [{epoch}/{args.epochs}] "
            f"Train Loss: {train_loss:.4f} Train {metric}: {train_acc:.4f} "
            f"Test {metric}: {test_acc:.4f} "
        )

print(f"Mean time per epoch: {torch.tensor(times).mean():.4f}s")
print(f"Total time: {sum(times):.4f}s")
print(f"Best test acc: {best_test_acc:.4f}")
