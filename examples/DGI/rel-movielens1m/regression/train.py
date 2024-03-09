# Deep Graph Infomax (DGI) for regression task in rel-movielens1m
# Paper: Deep Graph Infomax (ICLR 2019)
# Arxiv: https://arxiv.org/abs/1809.10341
# Test MSE Loss: 1.3718
# Runtime: 1737.6395s on a 6GB GPU (NVIDIA GeForce RTX 3060 laptop GPU)
# Cost: N/A
# Usage: python train.py

import sys
sys.path.append("../../../../examples/DGI/")
sys.path.append("../../../../rllm/dataloader/")

# import math
import torch
import torch.nn as nn
import numpy as np
import argparse
import time

# from utils import load_data function
from load_data import load_data
from examples.DGI.models import DGI, Model

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no_cuda', action='store_true',
                    default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=72, help='Random seed.')
parser.add_argument('--pre_epochs', type=int, default=60,
                    help='Number of epochs to pre_train.')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=0.0,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--gcn_lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--gcn_weight_decay', type=float, default=5e-4,
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
data, adj, features, labels, idx_train, idx_val, idx_test = load_data('movielens-regression', device=device)
# sp_adj = process.load_data_adj_cora(device)
nb_nodes = features.shape[0]
ft_size = features.shape[1]

features = features[np.newaxis]
sp_adj = adj

# Model and optimizer
model = DGI(ft_size, args.hidden, args.nonlinearity).to(device)
optimizer = torch.optim.Adam(model.parameters(),
                             lr=args.lr, weight_decay=args.weight_decay)

reg_model = Model(args.hidden, args.hidden).to(device)
opt = torch.optim.Adam(reg_model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

b_xent = nn.BCEWithLogitsLoss()
xent = nn.CrossEntropyLoss()
loss_func = nn.MSELoss()


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
    return embeds[0]


def train(epoch):
    t = time.time()
    reg_model.train()
    opt.zero_grad()
    output = reg_model(embeds, adj)
    loss_train = loss_func(output[idx_train], labels[idx_train])
    loss_train.backward()
    opt.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        reg_model.eval()
        output = reg_model(embeds, adj)

    loss_val = loss_func(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'time: {:.4f}s'.format(time.time() - t))


def test():
    reg_model.eval()
    output = reg_model(embeds, adj)
    loss_test = loss_func(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()))


# Train model
t_total = time.time()
# args.pre_epochs
for pre_epoch in range(args.pre_epochs):
    pre_train(pre_epoch)
print("Pre_train Finished!")
embeds = get_embed()
for epoch in range(args.epochs):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()
