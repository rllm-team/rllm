# SGC for regression task in rel-movielens1M
# Paper: Simplifying Graph Convolutional Networks. arXiv:1902.07153
# Test MSE Loss: 1.3562
# Runtime: 19.1168s (on a single 32G GPU)
# Cost: N/A
# Description: Simply apply SGC to movielens.
# Movies are linked iff a certain number of users rate them samely.
# Features were llm embeddings from table data to vectors.


from __future__ import division
from __future__ import print_function

import sys

from random import random
import time
import argparse
import numpy as np

# import math
import torch
import torch.nn as nn
import torch.optim as optim
# from sklearn.metrics import f1_score

from models import Model


path = "../../../../rllm/dataloader"
sys.path.append(path)
from load_data import load_data

t_total = time.time()
# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = 'cuda' if args.cuda else 'cpu'

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
data, adj, features, labels, \
    idx_train, idx_val, idx_test \
    = load_data('movielens-regression', device=device)


adj_drop = torch.zeros_like(adj.to_dense())
for i in range(adj.indices().shape[1]):
    if random() < 0.01:
        adj_drop[adj.indices()[0][i], adj.indices()[1][i]] = 1
adj_drop = adj_drop.to_sparse().coalesce()

# Model and optimizer
model = Model(nfeat=features.shape[1],
              nhid=args.hidden).to(device)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)


loss_func = nn.MSELoss()


def test():
    model.eval()
    output = model(features, adj, adj_drop)
    loss_test = loss_func(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss = {:.4f}".format(loss_test.item()))


# Train model
# t_total = time.time()
best_loss = 10000
best_epoch = 0
early_stop = 50

for epoch in range(args.epochs):
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

    if loss_val < best_loss:
        best_loss = loss_val
        best_epoch = epoch

    if epoch > best_epoch + early_stop:
        print('BREAK. There is no improvment for {} steps'.format(early_stop))
        break

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()
