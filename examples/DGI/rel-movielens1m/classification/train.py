# Deep Graph Infomax (DGI) for classification task in rel-movielens1m
# Paper: Deep Graph Infomax (ICLR 2019)
# Arxiv: https://arxiv.org/abs/1809.10341
# Test f1_score micro: 0.2512, macro: 0.1295
# Runtime: 63.0053s on a 6GB GPU (NVIDIA GeForce RTX 3060 laptop GPU)
# Cost: N/A
# Usage: python train.py
# Description: Paper Reproduction. Simply apply Deep Graph Infomax to Cora.
#   * Deep Graph Infomax (DGI) is a general approach for learning
#   node representations in an unsupervised manner.
#   * DGI gets node representations and the Logistic regression
#   model trains on this data.

import sys
sys.path.append("../../../../examples/DGI/")
sys.path.append("../../../../rllm/dataloader/")

# import math
import torch
import torch.nn as nn
import numpy as np
import argparse
import time
from sklearn.metrics import f1_score

# from utils import load_data function
from load_data import load_data
from examples.DGI.models import DGI, LogReg


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no_cuda', action='store_true',
                    default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=72, help='Random seed.')
parser.add_argument('--pre_epochs', type=int, default=200,
                    help='Number of epochs to pre_train.')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=0.0,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=512,
                    help='Number of hidden units.')
parser.add_argument('--batch_size', type=int, default=1,
                    help='Size of batch.')
parser.add_argument('--sparse', action='store_false',
                    default=True, help='If adjacency matrix.')
parser.add_argument('--nonlinearity', type=str, default='prelu',
                    help='Nonlinearity function to use.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = 'cuda' if args.cuda else 'cpu'


np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    print("cuda")

# Load data
data, adj, features, labels, idx_train, idx_val, idx_test = load_data('movielens-classification', device=device)
# sp_adj = process.load_data_adj_cora(device)
nb_nodes = features.shape[0]
ft_size = features.shape[1]
nb_classes = labels.shape[1]
features = features[np.newaxis]
labels_train = labels.cpu()[idx_train.cpu()]
labels_val = labels.cpu()[idx_val.cpu()]
labels_test = labels.cpu()[idx_test.cpu()]
labels = labels[np.newaxis]
sp_adj = adj

# Model and optimizer
model = DGI(ft_size, args.hidden, args.nonlinearity).to(device)
optimizer = torch.optim.Adam(model.parameters(),
                             lr=args.lr, weight_decay=args.weight_decay)

log = LogReg(args.hidden, nb_classes).to(device)
opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=0.0)


b_xent = nn.BCEWithLogitsLoss()
xent = nn.CrossEntropyLoss()


def pre_train(pre_epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    # The input eigenmatrix is shuffled
    idx = np.random.permutation(nb_nodes)
    shuf_fts = features[:, idx, :]
    lbl_1 = torch.ones(args.batch_size, nb_nodes)
    lbl_2 = torch.zeros(args.batch_size, nb_nodes)
    lbl = torch.cat((lbl_1, lbl_2), 1)

    if torch.cuda.is_available():
        shuf_fts = shuf_fts.cuda()
        lbl = lbl.cuda()

    output = model(features, shuf_fts, sp_adj if args.sparse else adj,
                   args.sparse, None, None, None)
    loss_train = b_xent(output, lbl)
    loss_train.backward()
    optimizer.step()

    print('pre_train epoch: {:04d}'.format(pre_epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'time: {:.4f}s'.format(time.time() - t))


def get_embed():
    model.eval()
    embeds, _ = model.embed(features, sp_adj if args.sparse else adj,
                            args.sparse, None)
    train_embs = embeds[0, idx_train]
    train_lbls = torch.argmax(labels[0, idx_train], dim=1)
    val_embs = embeds[0, idx_val]
    val_lbls = torch.argmax(labels[0, idx_val], dim=1)
    test_embs = embeds[0, idx_test]
    test_lbls = torch.argmax(labels[0, idx_test], dim=1)
    return train_embs, train_lbls, val_embs, val_lbls, test_embs, test_lbls


def train(epoch):
    t = time.time()
    log.train()
    opt.zero_grad()
    logits = log(train_embs)
    pred = np.where(logits.cpu() > -1.0, 1, 0)

    loss_train = xent(logits, train_lbls)
    loss_train.backward()
    f1_micro_train = f1_score(labels_train, pred, average="micro")
    f1_macro_train = f1_score(labels_train, pred, average="macro")
    opt.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        log.eval()
        logits = log(val_embs)

    loss_val = xent(logits, val_lbls)
    f1_micro_val = f1_score(labels_val, pred, average="micro")
    f1_macro_val = f1_score(labels_val, pred, average="macro")

    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'f1_train: {:.4f} {:.4f}'.format(f1_micro_train, f1_macro_train),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'f1_val: {:.4f} {:.4f}'.format(f1_micro_val, f1_macro_val),
          'time: {:.4f}s'.format(time.time() - t))


def test():
    log.eval()
    logits = log(test_embs)
    pred = np.where(logits.cpu() > -1.0, 1, 0)
    loss_test = xent(logits, test_lbls)
    f1_micro_test = f1_score(labels_test, pred, average="micro")
    f1_macro_test = f1_score(labels_test, pred, average="macro")
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "f1_test= {:.4f} {:.4f}".format(f1_micro_test, f1_macro_test))


# Train model
t_total = time.time()
for pre_epoch in range(args.pre_epochs):
    pre_train(pre_epoch)
print("Pre_train Finished!")
train_embs, train_lbls, val_embs, val_lbls, test_embs, test_lbls = get_embed()
for epoch in range(args.epochs):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()
