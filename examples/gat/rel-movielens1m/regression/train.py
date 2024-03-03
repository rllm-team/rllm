# Naive GAT for regression task in rel-movielens1M
# Paper: P Veličković, G Cucurull, A Casanova, A Romero, P Lio, Y Bengio (2017). Graph attention networks arXiv preprint arXiv:1710.10903
# Test MSE Loss: 1.3203
# Runtime: 411.4200s on a 8GB GPU (NVIDIA(R) GeForce RTX(TM) 3060Ti) epoch 50
# Cost: N/A
# Description: Simply apply GAT to movielens. Graph was obtained by sampling from foreign keys. Features were llm embeddings from table data to vectors.
from __future__ import division, print_function
import sys
sys.path.append("../../../../rllm/dataloader")
sys.path.append("../../../../examples/gat")
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import glob
from random import random
from models import Model
from load_data import load_data


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true',
                    default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true',
                    default=False, help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=72, help='Random seed.')
parser.add_argument('--epochs', type=int, default=10000,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=8,
                    help='Number of hidden units.')
parser.add_argument('--nb_heads', type=int, default=8,
                    help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.6,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2,
                    help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=100, help='Patience')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = 'cuda' if args.cuda else 'cpu'

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
data, adj, features, labels, idx_train, idx_val, idx_test = load_data(
    'movielens-regression')
adj_drop = torch.zeros_like(adj.to_dense()).to(device)
for i in range(adj.indices().shape[1]):
    if random() < 0.01:
        adj_drop[adj.indices()[0][i], adj.indices()[1][i]] = 1
adj_drop = adj_drop.to_sparse().coalesce()

if args.cuda:
    adj_drop = adj_drop.cuda()
# Model and optimizer

model = Model(nfeat=features.shape[1],
              nhid=args.hidden,
              dropout=args.dropout,
              nheads=args.nb_heads,
              alpha=args.alpha).to(device)

optimizer = optim.Adam(model.parameters(),
                       lr=args.lr,
                       weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

loss_func = nn.MSELoss()


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj, adj_drop)
    loss_train = loss_func(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()
    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj, adj_drop)
    loss_val = loss_func(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'time: {:.4f}s'.format(time.time() - t))
    return loss_val.data.item()


def compute_test():
    model.eval()
    output = model(features, adj, adj_drop)
    loss_test = loss_func(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()))


# Train model
t_total = time.time()
loss_values = []
bad_counter = 0
best = args.epochs + 1
best_epoch = 0
for epoch in range(args.epochs):
    loss_values.append(train(epoch))
    torch.save(model.state_dict(), '{}.pkl'.format(epoch))
    if loss_values[-1] < best:
        best = loss_values[-1]
        best_epoch = epoch
        bad_counter = 0
    else:
        bad_counter += 1

    if bad_counter == args.patience:
        break

    files = glob.glob('*.pkl')
    for file in files:
        epoch_nb = int(file.split('.')[0])
        if epoch_nb < best_epoch:
            os.remove(file)

files = glob.glob('*.pkl')
for file in files:
    epoch_nb = int(file.split('.')[0])
    if epoch_nb > best_epoch:
        os.remove(file)

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Restore best model
print('Loading {}th epoch'.format(best_epoch))
model.load_state_dict(torch.load('{}.pkl'.format(best_epoch)))

# Testing
compute_test()
os.remove("{}.pkl".format(best_epoch))
