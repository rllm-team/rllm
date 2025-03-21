# The OGC method from the "From Cluster Assumption to Graph Convolution:
# Graph-based Semi-Supervised Learning Revisited" paper.
# ArXiv: https://arxiv.org/abs/2309.13599

# Datasets  CiteSeer    Cora      PubMed
# Acc       0.773       0.869     0.837
# Time      3.7s        2.3s      4.3s

import argparse
import time
import sys
import os.path as osp

import torch
import torch.nn.functional as F
from torch import Tensor

sys.path.append("./")
sys.path.append("../")
from rllm.data import GraphData
from rllm.datasets import PlanetoidDataset
from rllm.transforms.graph_transforms import GCNTransform
from rllm.nn.conv.graph_conv import LGCConv

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset", type=str, default="cora", choices=["citeseer, cora, pubmed"]
)
parser.add_argument("--decline", type=float, default=0.9, help="decline rate"),
parser.add_argument("--lr_sup", type=float, default=0.001, help="lr for loss")
parser.add_argument("--lr_W", type=float, default=0.5, help="lr for W")
parser.add_argument("--beta", type=float, default=0.1, help="moving probability")
parser.add_argument("--sim_tol", type=float, default=0.995, help="max similarity")
parser.add_argument("--patience", type=int, default=2, help="tolerance")
args = parser.parse_args()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
path = osp.join(osp.dirname(osp.realpath(__file__)), "..", "data")
data = PlanetoidDataset(path, args.dataset, force_reload=True)[0]

# Transform data
transform = GCNTransform()
data = transform(data).to(device)

# One-hot encoding for labels
y_one_hot = F.one_hot(input=data.y, num_classes=data.num_classes).float()
data.trainval_mask = data.train_mask | data.val_mask

# LIM trick, else use trainval_mask to construct S
S = torch.diag(data.train_mask).float().to_sparse(layout=torch.sparse_coo)
I_N = torch.eye(data.num_nodes).to_sparse(layout=torch.sparse_coo).to(device)

# Lazy random walk (also known as lazy graph convolution):
# lazy_adj = args.beta * data.adj + (1 - args.beta) * I_N
lr_sup = args.lr_sup


# Define model
class LinearNeuralNetwork(torch.nn.Module):
    def __init__(self, num_feats: int, num_classes: int, bias: bool = True):
        super().__init__()
        self.W = torch.nn.Linear(num_feats, num_classes, bias=bias)
        self.conv = LGCConv(beta=args.beta)

    def forward(self, x: Tensor) -> Tensor:
        return self.W(x)

    @torch.no_grad()
    def test(self, U: Tensor, y_one_hot: Tensor, data: GraphData):
        self.eval()
        out = self(U)

        loss = F.mse_loss(
            out[data.trainval_mask],
            y_one_hot[data.trainval_mask],
        )

        accs = []
        pred = out.argmax(dim=-1)
        for mask in [data.trainval_mask, data.test_mask]:
            accs.append(float((pred[mask] == data.y[mask]).sum() / mask.sum()))

        return float(loss), accs[0], accs[1], pred

    def update_W(self, U: Tensor, y_one_hot: Tensor, data: GraphData):
        optimizer = torch.optim.SGD(self.parameters(), lr=args.lr_W)
        self.train()
        optimizer.zero_grad()
        pred = self(U)
        loss = F.mse_loss(
            pred[data.trainval_mask], y_one_hot[data.trainval_mask,], reduction="sum"
        )
        loss.backward()
        optimizer.step()
        return self(U).data

    def update_U(self, U: Tensor, y_one_hot: Tensor, pred: Tensor, adj: Tensor):
        global lr_sup

        # Update the smoothness loss via LGC:
        U = self.conv(U, adj)

        # Update the supervised loss via SEB:
        dU_sup = 2 * (S @ (-y_one_hot + pred)) @ self.W.weight.data
        U = U - lr_sup * dU_sup

        lr_sup = lr_sup * args.decline
        return U


# Set up model
model = LinearNeuralNetwork(
    num_feats=data.x.shape[1],
    num_classes=data.num_classes,
    bias=False,
).to(device)


def ogc() -> float:
    U = data.x
    _, _, last_acc, last_pred = model.test(U, y_one_hot, data)

    patience = 0
    for i in range(1, 65):
        # Updating W by training a simple linear neural network:
        pred = model.update_W(U, y_one_hot, data)

        # Updating U by LGC and SEB jointly:
        U = model.update_U(U, y_one_hot, pred, data.adj)

        loss, trainval_acc, test_acc, pred = model.test(U, y_one_hot, data)
        print(
            f"Epoch: {i:02d}, Loss: {loss:.4f}, "
            f"Train+Val Acc: {trainval_acc:.4f} Test Acc {test_acc:.4f}"
        )

        sim_rate = float((pred == last_pred).sum()) / pred.size(0)
        if sim_rate > args.sim_tol:
            patience += 1
            if patience > args.patience:
                break

        last_acc, last_pred = test_acc, pred

    return last_acc


start_time = time.time()
test_acc = ogc()
print(f"Total Time: {time.time() - start_time:.4f}s")
print(f"Test Accuracy: {test_acc:.4f}")
