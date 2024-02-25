# OGC for classification task in Cora
# Paper: Wang, Z., Ding, H., Pan, L., Li, J., Gong, Z., & Yu, P. S. (2023). From Cluster Assumption to Graph Convolution: Graph-based Semi-Supervised Learning Revisited. ArXiv. /abs/2309.13599
# Test Accuracy: 0.8660
# Runtime: 6.3392s on a single CPU (Intel(R) Core(TM) i5-10210U CPU @ 1.60GHz 2.11 GHz)
# Cost: N/A
# Description: Paper Reproduction. Simply apply OGC to Cora.

import sys 
sys.path.append("../../../rllm/dataloader")

import argparse
import time
import warnings
import torch
import torch.nn.functional as F
from torch import Tensor

from load_data import load_data

warnings.filterwarnings('ignore', '.*Sparse CSR tensor support.*')

start_time = time.time()

decline = 0.9  # decline rate
eta_sup = 0.001  # learning rate for supervised loss
eta_W = 0.5  # learning rate for updating W
beta = 0.1  # moving probability that a node moves to neighbors
max_sim_tol = 0.995  # max label prediction similarity between iterations
max_patience = 2  # tolerance for consecutive similar test predictions

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Cora')
args = parser.parse_args()

data, adj, feature, label, idx_train, idx_val, idx_test = load_data('cora')
num_nodes = feature.shape[0]

adj = adj.to_dense()
y = torch.max(label, dim=1)[1]
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


class LinearNeuralNetwork(torch.nn.Module):
    def __init__(self, num_features: int, num_classes: int, bias: bool = True):
        super().__init__()
        self.W = torch.nn.Linear(num_features, num_classes, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.W(x)

    @torch.no_grad()
    def test(self, U: Tensor, label: Tensor, data):
        self.eval()
        out = self(U)

        loss = F.mse_loss(
            out[trainval_mask],
            label[trainval_mask],
        )

        accs = []
        pred = out.argmax(dim=-1)
        for mask in (trainval_mask, test_mask):
            accs.append(float((pred[mask] == y[mask]).sum() / mask.sum()))

        return float(loss), accs[0], accs[1], pred

    def update_W(self, U: Tensor, label: Tensor, data):
        optimizer = torch.optim.SGD(self.parameters(), lr=eta_W)
        self.train()
        optimizer.zero_grad()
        pred = self(U)
        loss = F.mse_loss(pred[trainval_mask], label[
            trainval_mask,
        ], reduction='sum')
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
    _, _, last_acc, last_pred = model.test(U, label, data)

    patience = 0
    for i in range(1, 65):
        # Updating W by training a simple linear neural network:
        pred, W = model.update_W(U, label, data)

        # Updating U by LGC and SEB jointly:
        U = update_U(U, label, pred, W)

        loss, trainval_acc, test_acc, pred = model.test(U, label, data)
        print(f'Epoch: {i:02d}, Loss: {loss:.4f}, '
              f'Train+Val Acc: {trainval_acc:.4f} Test Acc {test_acc:.4f}')

        sim_rate = float((pred == last_pred).sum()) / pred.size(0)
        if (sim_rate > max_sim_tol):
            patience += 1
            if (patience > max_patience):
                break

        last_acc, last_pred = test_acc, pred

    return last_acc


test_acc = ogc()
print(f'Test Accuracy: {test_acc:.4f}')
print(f'Total Time: {time.time() - start_time:.4f}s')
