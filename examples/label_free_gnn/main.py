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

import torch
import torch.nn.functional as F
from langchain_community.llms import LlamaCpp
from numpy import mean
from tqdm import tqdm

sys.path.append("../")

import rllm.transforms as T
from annotation.annotation import annotate
from node_selection.node_selection import active_generate_mask, post_filter
from rllm.datasets.tagdataset import TAGDataset
from rllm.llm.llm_module.langchain_llm import LangChainLLM
from rllm.nn.conv.graph_conv import GCNConv

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    type=str,
    default="cora",
    choices=["cora", "citeseer", "pubmed"],
    help="dataset name",
)
parser.add_argument(
    "--hidden_channels", type=int, default=64, help="number of hidden channels in GCN"
)
parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
parser.add_argument("--wd", type=float, default=5e-4, help="weight decay")
parser.add_argument("--epochs", type=int, default=150, help="number of training epochs")
parser.add_argument("--use_cache", type=bool, default=True, help="whether to use cache")
parser.add_argument(
    "--active_method",
    type=str,
    default="Random",
    choices=["Random", "VertexCover", "FeatProp"],
    help="method for active node selection",
)
parser.add_argument(
    "--post_filter", type=bool, default=False, help="whether to perform post filtering"
)
parser.add_argument(
    "--filter_strategy",
    type=str,
    default="conf+density",
    choices=["conf+density", "conf+density+entropy"],
    help="strategy for post filtering",
)
parser.add_argument(
    "--weighted_loss", type=bool, default=False, help="whether to use weighted loss"
)
parser.add_argument(
    "--n_tries", type=int, default=3, help="number of tries when asking LLM"
)
parser.add_argument(
    "--budget", type=int, default=20, help="number of LLM queries per class"
)
parser.add_argument(
    "--val", type=bool, default=False, help="whether to use validation set"
)
parser.add_argument("--n_rounds", type=int, default=20, help="number of rounds")
args = parser.parse_args()


path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data")

transform = T.Compose([T.NormalizeFeatures("l2"), T.GCNNorm()])
dataset = TAGDataset(
    path, args.dataset, use_cache=args.use_cache, transform=transform, force_reload=True
)
data = dataset[0]

if not args.use_cache:
    model_path = "/path/to/llm"
    llm = LangChainLLM(LlamaCpp(model_path=model_path, n_gpu_layers=33))


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, adj):
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.conv1(x, adj))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, adj)
        return x


class Trainer:
    def __init__(self, data, model, optimizer, masks, epochs, val, weighted_loss):
        self.data = data
        self.model = model
        self.optimizer = optimizer
        self.train_mask = masks["train_mask"]
        self.val_mask = masks["val_mask"]
        self.test_mask = masks["test_mask"]
        self.epochs = epochs
        self.val = val
        self.weighted_loss = weighted_loss
        self.losses = []

    def train(self):
        best_val_acc = 0
        best_test_acc = 0
        for epoch in tqdm(range(1, self.epochs + 1)):
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
            self.losses.append(loss.item())
            if self.val:
                train_acc, val_acc, test_acc = trainer.test()
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_test_acc = test_acc
            else:
                train_acc, test_acc = trainer.test()
                if test_acc > best_test_acc:
                    best_test_acc = test_acc
        return best_test_acc

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

method = args.active_method
if args.post_filter:
    method = "PS-" + method
if args.weighted_loss:
    method += "-W"

print(f"using dataset: {args.dataset}, method: {method}")

for i in range(args.n_rounds):
    train_mask, val_mask, test_mask = active_generate_mask(
        data, method=args.active_method, val=args.val, budget=args.budget
    )
    if not args.use_cache:
        pl_indices = torch.nonzero(train_mask | val_mask, as_tuple=False).squeeze()
        data = annotate(data, pl_indices, llm, args.n_tries)
    if args.post_filter:
        filtered_mask = post_filter(data, train_mask | val_mask, args.filter_strategy)
        train_mask = train_mask & filtered_mask
        val_mask = val_mask & filtered_mask

    model = GCN(
        in_channels=data.x.shape[1],
        hidden_channels=args.hidden_channels,
        out_channels=data.num_classes,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    masks = {"train_mask": train_mask, "val_mask": val_mask, "test_mask": test_mask}

    trainer = Trainer(
        data, model, optimizer, masks, args.epochs, args.val, args.weighted_loss
    )
    best_test_acc = trainer.train()
    acc_list.append(best_test_acc)
    print(f"round {i} best test acc: {best_test_acc:.4f}")


print(f"dataset: {args.dataset}, method: {method}, mean accuracy: {mean(acc_list)}")
