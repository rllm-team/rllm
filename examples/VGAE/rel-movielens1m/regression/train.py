# Naive GAE for classification task in rel-movielens1M
# Paper: T. N. Kipf, M. Welling, Variational Graph Auto-Encoders ArXiv:1611.07308
# Test MSE Loss: 1.9453
# Runtime: 116.94
# Cost: N/A
# Description: Simply apply GAE to movielens. Graph was obtained by sampling from foreign keys. Features were llm embeddings from table data to vectors.

from __future__ import division
from __future__ import print_function
import torch
import argparse
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
from torch import optim
from model import GAE_REGRESSION
import sys
sys.path.append("../..")
from utils import sparse_mx_to_torch_sparse_tensor, preprocess_graph, adj_matrix_to_list, change_to_matrix, add_self_loops, combined_regression_loss
sys.path.append("../../../../rllm/dataloader")
from load_data import load_data
import time

time_start = time.time()
# Define command-line arguments using argparse
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=50,
                    help='Random seed.')
parser.add_argument('--epochs', type=int, default=50,
                    help='Number of epochs to train.')
parser.add_argument('--hidden1', type=int, default=16,
                    help='Number of units in hidden layer 1.')
parser.add_argument('--hidden2', type=int, default=8,
                    help='Number of units in hidden layer 2.')
parser.add_argument('--lr', type=float, default=5e-3,
                    help='Initial learning rate.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--weight_decay', type=float, default=1e-5,
                    help='Weight decay (L2 loss on parameters).')

args = parser.parse_args()

def gae_for(args):
    print("Using {} dataset".format("movielens-regression"))
    # load movielens-regression dataset
    data, adj, features, labels, idx_train, idx_val, idx_test = load_data('movielens-regression')
    # print("adj_shape", adj.shape)
    # print("labels_shape", labels.shape)
    n_nodes, feat_dim = features.shape
    # Some preprocessing
    adj_matrix = change_to_matrix(adj)
    adj_train = add_self_loops(adj_matrix)
    adj = adj_train
    adj_label = adj_train + sp.eye(adj_train.shape[0])
    adj_label = torch.FloatTensor(adj_label.toarray())
    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
    adj_norm = preprocess_graph(adj)
    num_classes = labels.shape[0]
    # print("adj_norm_shape", adj_norm.shape)
    # build the GAE_REGRESSION model and optimizer
    # model = GAE_REGRESSION(feat_dim, args.hidden1, args.hidden2, args.dropout)
    model = GAE_REGRESSION(feat_dim, args.hidden1, args.hidden2, args.dropout)
    # optimizer = optim.SGD(model.parameters(), lr=args.lr)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lossf = nn.MSELoss()
    # training loop
    for epoch in range(args.epochs):
        # print(epoch)
        t = time.time()
        model.train()
        optimizer.zero_grad()
        recovered, aaa, mu, logvar = model(features, adj_norm)
        # recovered = model(features, adj_norm)
        loss = combined_regression_loss(preds=aaa, labels=adj_label,
                                            mu=mu, logvar=logvar, n_nodes=n_nodes,
                                            norm=norm, pos_weight=pos_weight, preds_logits=recovered[idx_train],
                                            labels_binary=labels[idx_train])
        loss.backward()
        # cur_loss = loss.item()
        optimizer.step()

        # evaluate on validation set
        loss_val = combined_regression_loss(preds=aaa, labels=adj_label,
                                            mu=mu, logvar=logvar, n_nodes=n_nodes,
                                            norm=norm, pos_weight=pos_weight, preds_logits=recovered[idx_val],
                                            labels_binary=labels[idx_val])
        print('Epoch: {:04d}'.format(epoch + 1),
              'loss_train: {:.4f}'.format(loss.item()),
              'loss_val: {:.4f}'.format(loss_val.item()),
              'time: {:.4f}s'.format(time.time() - t))
    print("Optimization Finished!")
    time_end = time.time()
    print("Total time elapsed:", time_end - time_start)
    # test the model
    model.eval()
    recovered, aaa, mu, logvar = model(features, adj_norm)
    # recovered = model(features, adj_norm)
    loss_test = lossf(recovered[idx_test], labels[idx_test])

    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item())
          )



if __name__ == '__main__':
    # fix random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    gae_for(args)
