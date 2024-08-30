# The GAT method from the
# "Graph Attention Networks" paper.
# ArXiv: https://arxiv.org/abs/1710.10903

# Datasets  CiteSeer    Cora      PubMed
# Acc       0.717       0.823     0.778
# Time      16.6s       8.4s      15.6s

import argparse
import os.path as osp
import time
import torch
import torch.nn.functional as F
import sys
sys.path.append('../')
import rllm.transforms as T
from rllm.datasets.planetoid import PlanetoidDataset
from rllm.nn.conv.gat_conv import GATConv

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cora',
                    choices=['citeseer', 'cora', 'pubmed'])
parser.add_argument('--hidden_channels', type=int, default=8)
parser.add_argument('--heads', type=int, default=8, help="Attention heads")
parser.add_argument('--lr', type=float, default=5e-3, help='Learning rate')
parser.add_argument('--wd', type=float, default=5e-4, help='Weight decay')
parser.add_argument('--epochs', type=int, default=200, help="Training epochs")
parser.add_argument('--dropout', type=float, default=0.6, help='Graph Dropout')
args = parser.parse_args()

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data')
dataset = PlanetoidDataset(path,
                           args.dataset,
                           transform=T.NormalizeFeatures('sum'))
data = dataset[0]
data.adj = torch.eye(data.adj.size(0)) + data.adj
indices = torch.nonzero(data.adj, as_tuple=False)
values = data.adj[indices[:, 0], indices[:, 1]]
sparse_tensor = torch.sparse_coo_tensor(indices.t(), values, data.adj.size())
data.adj = sparse_tensor


class GAT(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 dropout: float = 0.,
                 heads: int = 8):
        super().__init__()
        self.dropout = dropout
        self.conv1 = GATConv(in_channels, hidden_channels, heads)
        self.conv2 = GATConv(hidden_channels*heads, out_channels,
                             heads=1, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, adj))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, adj)
        return x


model = GAT(
    in_channels=data.x.shape[1],
    hidden_channels=args.hidden_channels,
    out_channels=data.num_classes,
    heads=args.heads,
    dropout=args.dropout,
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=args.lr,
    weight_decay=args.wd
)  # Only perform weight-decay on first convolution.
loss_fn = torch.nn.CrossEntropyLoss()


def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.adj)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def test():
    model.eval()
    out = model(data.x, data.adj)
    pred = out.argmax(dim=1)

    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        correct = float(pred[mask].eq(data.y[mask]).sum().item())
        accs.append(correct / int(mask.sum()))
    return accs


best_val_acc = best_test_acc = 0
times = []
for epoch in range(1, args.epochs + 1):
    start = time.time()
    train_loss = train()
    train_acc, val_acc, test_acc = test()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_test_acc = test_acc
    print(
        f"Epoch: [{epoch}/{args.epochs}]"
        f"Loss: {train_loss:.4f} train_acc: {train_acc:.4f} "
        f"val_acc: {val_acc:.4f} test_acc: {test_acc:.4f} "
    )
    times.append(time.time() - start)
print(f'Mean time per epoch: {torch.tensor(times).mean():.4f}s')
print(f'Best test acc: {best_test_acc:.4f}')
