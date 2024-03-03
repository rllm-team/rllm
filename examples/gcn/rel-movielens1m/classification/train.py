# Naive GCN for classification task in rel-movielens1M
# Paper: Kipf, T. N., & Welling, M. (2016). Semi-Supervised Classification with Graph Convolutional Networks. ArXiv. /abs/1609.02907
# Test f1_score micro: 0.3911, macro: 0.0756
# Runtime: 36.2197s on a single CPU (Intel(R) Core(TM) i5-10210U CPU @ 1.60GHz 2.11 GHz)
# Cost: N/A
# Description: Simply apply GCN to movielens. Movies are linked iff a certain number of users rate them samely. Features were llm embeddings from table data to vectors.

# Comment: Over-smoothing is significant.

from __future__ import division
from __future__ import print_function

import sys 
sys.path.append("../../../../rllm/dataloader/")

import time
import argparse
import numpy as np

# import math
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score

# from utils import load_data

from load_data import load_data
from models import GCN

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
data, adj, features, labels, idx_train, idx_val, idx_test = load_data('movielens-classification', device=device)
labels_train = labels.cpu()[idx_train.cpu()]
labels_val = labels.cpu()[idx_val.cpu()]
labels_test = labels.cpu()[idx_test.cpu()]

# Model and optimizer
model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.shape[1],
            dropout=args.dropout).to(device)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)
# if args.cuda:
#     model.cuda()

loss_func = nn.BCEWithLogitsLoss()

def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    pred = np.where(output.cpu() > -1.0, 1, 0)
    # print('output[0] =', output[1], output[15])
    # print('f1 =', f1_score(labels[idx_train], pred[idx_train], average=None))

    loss_train = loss_func(output[idx_train], labels[idx_train])
    f1_micro_train = f1_score(labels_train, pred[idx_train.cpu()], average="micro")
    f1_macro_train = f1_score(labels_train, pred[idx_train.cpu()], average="macro")
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    loss_val = loss_func(output[idx_val], labels[idx_val])
    f1_micro_val = f1_score(labels_val, pred[idx_val.cpu()], average="micro")
    f1_macro_val = f1_score(labels_val, pred[idx_val.cpu()], average="macro")
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'f1_train: {:.4f} {:.4f}'.format(f1_micro_train, f1_macro_train),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'f1_val: {:.4f} {:.4f}'.format(f1_micro_val, f1_macro_val),
          'time: {:.4f}s'.format(time.time() - t))


def test():
    model.eval()
    output = model(features, adj)
    pred = np.where(output.cpu() > -1.0, 1, 0)
    loss_test = loss_func(output[idx_test], labels[idx_test])
    f1_micro_test = f1_score(labels_test, pred[idx_test.cpu()], average="micro")
    f1_macro_test = f1_score(labels_test, pred[idx_test.cpu()], average="macro")
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "f1_test= {:.4f} {:.4f}".format(f1_micro_test, f1_macro_test))


# Train model
for epoch in range(args.epochs):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()
