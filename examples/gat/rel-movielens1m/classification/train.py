# Naive GAT for classification task in rel-movielens1M
# Paper: P Veličković, G Cucurull, A Casanova, A Romero, P Lio, Y Bengio (2017). Graph attention networks arXiv preprint arXiv:1710.10903
# Test f1_score micro: 0.3934, macro: 0.0585
# Runtime: 69.7902s on a single CPU (Intel(R) Core(TM) i5-12400F CPU @ 2.50GHz 4.40GHz) epoch 200
# Cost: N/A
# Description: Simply apply GAT to movielens. Movies are linked iff a certain number of users rate them samely. Features were llm embeddings from table data to vectors.
from __future__ import division, print_function
import sys
sys.path.append("../../../../rllm/dataloader")
sys.path.append("../../../../examples/gat")
from load_data import load_data
from models import GATClassification

import os
import glob
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from sklearn.metrics import f1_score


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true',
                    default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true',
                    default=False, help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--nb_heads', type=int, default=2,
                    help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2,
                    help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=100, help='Patience')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
data, adj, features, labels, idx_train, idx_val, idx_test = load_data(
    'movielens-classification')
labels_train = labels.cpu()[idx_train.cpu()]
labels_val = labels.cpu()[idx_val.cpu()]
labels_test = labels.cpu()[idx_test.cpu()]

# Model and optimizer
model = GATClassification(nfeat=features.shape[1],
                          nhid=args.hidden,
                          nclass=labels.shape[1],
                          dropout=args.dropout,
                          nheads=args.nb_heads,
                          alpha=args.alpha)
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

features, adj, labels = Variable(features), Variable(adj), Variable(labels)
loss_func = nn.BCEWithLogitsLoss()


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    pred = np.where(output.cpu() > -1.0, 1, 0)
    loss_train = loss_func(output[idx_train], labels[idx_train])
    f1_micro_train = f1_score(
        labels_train, pred[idx_train.cpu()], average="micro")
    f1_macro_train = f1_score(
        labels_train, pred[idx_train.cpu()], average="macro")
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

    return loss_val.data.item()


def compute_test():
    model.eval()
    output = model(features, adj)
    pred = np.where(output.cpu() > -1, 1, 0)
    loss_test = loss_func(output[idx_test], labels[idx_test])
    f1_micro_test = f1_score(labels_test, pred[idx_test.cpu()],
                             average="micro")
    f1_macro_test = f1_score(labels_test, pred[idx_test.cpu()],
                             average="macro")
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "f1_test= {:.4f} {:.4f}".format(f1_micro_test, f1_macro_test))


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
