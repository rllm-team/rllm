# Naive OGC for classification task in rel-movielens1M
# Paper: Wang, Z., Ding, H., Pan, L., Li, J., Gong, Z., & Yu, P. S. (2023). From Cluster Assumption to Graph Convolution: Graph-based Semi-Supervised Learning Revisited. ArXiv. /abs/2309.13599
# Test f1_score micro: 0.1696 macro: 0.1515
# Runtime: 7.1720s on a single CPU (Intel(R) Core(TM) i5-10210U CPU @ 1.60GHz 2.11 GHz)
# Cost: N/A
# Description: Simply apply OGC to movielens. Movies are linked iff a certain number of users rate them samely. Features were llm embeddings from table data to vectors.

import sys 
sys.path.append("../../../../rllm/dataloader")
import argparse
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from sklearn.metrics import f1_score

from load_data import load_data

warnings.filterwarnings('ignore', '.*Sparse CSR tensor support.*')

decline = 0.9  # decline rate
eta_sup = 0.001  # learning rate for supervised loss
eta_W = 0.5  # learning rate for updating W
beta = 0.1  # moving probability that a node moves to neighbors
max_sim_tol = 0.995  # max label prediction similarity between iterations
max_patience = 2  # tolerance for consecutive similar test predictions

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Cora')
args = parser.parse_args()

data, adj, feature, label, idx_train, idx_val, idx_test = load_data('movielens-classification')
num_nodes = feature.shape[0]

feature /= feature.sum(dim=1).view(-1, 1)
deg = adj.to_dense().sum(dim=1) ** (-0.5)
deg = torch.where(torch.isinf(deg), torch.full_like(deg, 0), deg)
deg = torch.diag(deg)
adj = torch.spmm(deg, torch.spmm(adj, deg)).to_sparse()
adj = adj.to_dense()

label = label.type(torch.FloatTensor)
train_mask = torch.BoolTensor([(i in idx_train) for i in range(num_nodes)])
val_mask = torch.BoolTensor([(i in idx_val) for i in range(num_nodes)])
test_mask = torch.BoolTensor([(i in idx_test) for i in range(num_nodes)])

trainval_mask = train_mask | val_mask
# LIM track, else use trainval_mask to construct S
S = torch.diag(train_mask).float().to_sparse()
I_N = torch.eye(num_nodes).to_sparse(layout=torch.sparse_csr)

# Lazy random walk (also known as lazy graph convolution):
lazy_adj = beta * adj + (1 - beta) * I_N

loss_func = nn.BCEWithLogitsLoss()

class LinearNeuralNetwork(torch.nn.Module):
    def __init__(self, num_features: int, num_classes: int, bias: bool = True):
        super().__init__()
        self.W = torch.nn.Linear(num_features, num_classes, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        return F.logsigmoid(self.W(x))

    @torch.no_grad()
    def test(self, U: Tensor, label: Tensor, data):
        self.eval()
        out = self(U)
        # print(out)

        loss = loss_func(
            out[trainval_mask],
            label[trainval_mask],
        )
        pred = np.where(out.cpu() > -1.0, 1, 0)
        f1_micro_train = f1_score(label[idx_train], pred[idx_train], average="micro")
        f1_macro_train = f1_score(label[idx_train], pred[idx_train], average="macro")

        return float(loss), f1_micro_train, f1_macro_train, pred

    def update_W(self, U: Tensor, label: Tensor, data):
        optimizer = torch.optim.SGD(self.parameters(), lr=eta_W)
        self.train()
        optimizer.zero_grad()
        pred = self(U)
        loss = loss_func(pred[trainval_mask], label[trainval_mask])
        loss.backward()
        optimizer.step()
        return self(U).data, self.W.weight.data


model = LinearNeuralNetwork(
    num_features=feature.shape[1],
    num_classes=label.shape[1],
    bias=False,
)


def update_U(U: Tensor, label: Tensor, pred: Tensor, W: Tensor):
    global eta_sup

    # Update the smoothness loss via LGC:
    U = lazy_adj @ U

    # Update the supervised loss via SEB:
    dU_sup = 2 * (S @ (-label + pred)) @ W
    U = U - eta_sup * dU_sup

    eta_sup = eta_sup * decline
    return U


def ogc() -> float:
    U = feature
    # _, _, last_acc, last_pred = model.test(U, label, data)

    patience = 0
    for i in range(1, 65):
        # Updating W by training a simple linear neural network:
        pred, W = model.update_W(U, label, data)

        # Updating U by LGC and SEB jointly:
        U = update_U(U, label, pred, W)

        loss, f1_micro, f1_macro, pred = model.test(U, label, data)
        print(f'Epoch: {i:02d}, Loss: {loss:.4f}, '
              f'f1_micro: {f1_micro:.4f} f1_macro: {f1_macro:.4f}')

        # sim_rate = float((pred == last_pred).sum()) / (pred.shape[0] * pred.shape[1])
        # if (sim_rate > max_sim_tol):
        #     patience += 1
        #     if (patience > max_patience):
        #         break

        if i == 64:
            return f1_micro, f1_macro


start_time = time.time()
f1_micro, f1_macro = ogc()
print(f'Test Accuracy: {f1_micro:.4f}, {f1_macro:.4f}')
print(f'Total Time: {time.time() - start_time:.4f}s')
