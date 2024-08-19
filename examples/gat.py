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
parser.add_argument('--dataset', type=str, default='cora')
parser.add_argument('--hidden_channels', type=int, default=16)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--wd', type=float, default=0.0)
parser.add_argument('--epochs', type=int, default=500)
args = parser.parse_args()

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data')
dataset = PlanetoidDataset(path, args.dataset, transform=T.NormalizeFeatures())
# data = dataset.item()
data = dataset[0]


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels)
        self.conv2 = GATConv(hidden_channels, out_channels)

    def forward(self, x, adj):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.relu(self.conv1(x, adj))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, adj)
        return x


model = GAT(
    in_channels=data.x.shape[1],
    hidden_channels=args.hidden_channels,
    out_channels=data.num_classes,
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
    loss = loss_fn(out[data.train_mask], data.y[data.train_mask])
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
    times.append(time.time() - start)
print(f'Mean time per epoch: {torch.tensor(times).mean():.4f}s')
print(f'Best test acc: {best_test_acc:.4f}')
