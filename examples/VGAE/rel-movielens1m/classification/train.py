# Naive GAE for classification task in rel-movielens1M
# Paper: T. N. Kipf, M. Welling, Variational Graph Auto-Encoders ArXiv:1611.07308
# Test f1_score micro: 0.2842, macro: 0.1267
# Runtime: 34.29
# Cost: N/A
# Description: Simply apply GAE to movielens. Movies are linked iff a certain number of users rate them samely. Features were llm embeddings from table data to vectors.

from __future__ import division
from __future__ import print_function

import argparse
import torch
import numpy as np
import scipy.sparse as sp
from torch import optim

from model import GAE_CLASSIFICATION
from sklearn.metrics import f1_score
import sys
sys.path.append("../../../../rllm/dataloader")
import time
from load_data import load_data
sys.path.append("../..")
from utils import sparse_mx_to_torch_sparse_tensor, preprocess_graph, adj_matrix_to_list, change_to_matrix, add_self_loops, combined_classification_loss

time_start = time.time()
# Define command-line arguments using argparse
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=50, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--hidden1', type=int, default=32, help='Number of units in hidden layer 1.')
parser.add_argument('--hidden2', type=int, default=16, help='Number of units in hidden layer 2.')
parser.add_argument('--lr', type=float, default=0.002, help='Initial learning rate.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')

args = parser.parse_args()


def gae_for(args):
    print("Using {} dataset".format("movielens-classification"))
    # load movielens-classification dataset
    data, adj, features, labels, idx_train, idx_val, idx_test = load_data('movielens-classification')
    n_nodes, feat_dim = features.shape
    # convert adj to networkx graph
    adj_matrix = change_to_matrix(adj)
    adj_train = add_self_loops(adj_matrix)
    adj = adj_train
    # Some preprocessing
    # adj_norm = preprocess_graph(adj)
    adj_label = adj_train + sp.eye(adj_train.shape[0])
    adj_label = torch.FloatTensor(adj_label.toarray())

    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
    adj_norm = preprocess_graph(adj)
    num_classes = labels.shape[1]

    # build the GAE_CLASSIFICATION model and optimizer
    model = GAE_CLASSIFICATION(feat_dim, args.hidden1, args.hidden2, num_classes, args.dropout)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    # training loop
    for epoch in range(args.epochs):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        recovered, aaa, mu, logvar = model(features, adj_norm)
        loss = combined_classification_loss(preds=aaa, labels=adj_label,
                             mu=mu, logvar=logvar, n_nodes=n_nodes,
                             norm=norm, pos_weight=pos_weight, preds_logits=recovered[idx_train], labels_binary=labels[idx_train])
        # loss = loss_f(recovered[idx_train], labels[idx_train])
        loss.backward()
        cur_loss = loss.item()
        optimizer.step()

        # evaluate on validation set
        pred_train = np.where(recovered[idx_val].detach().numpy() > -1.0, 1, 0)
        f1_micro_train = f1_score(labels[idx_val].detach().numpy(), pred_train, average="micro")
        f1_macro_train = f1_score(labels[idx_val].detach().numpy(), pred_train, average="macro")
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(cur_loss),
              'f1_val_micro: {:.4f}'.format(f1_micro_train),
              'f1_val_macro: {:.4f}'.format(f1_macro_train),
              "time=", "{:.5f}".format(time.time() - t)
              )

    print("Optimization Finished!")
    time_end = time.time()
    print("Total time elapsed:", time_end - time_start)
    # test the model
    model.eval()
    recovered, aaa, mu, logvar = model(features, adj_norm)
    loss_test = combined_classification_loss(preds=aaa, labels=adj_label,
                             mu=mu, logvar=logvar, n_nodes=n_nodes,
                             norm=norm, pos_weight=pos_weight, preds_logits=recovered[idx_test], labels_binary=labels[idx_test])
    # Calculate F1 scores for the test set
    pred_test = np.where(recovered[idx_test].detach().numpy() > -1.0, 1, 0)
    f1_micro_test = f1_score(labels[idx_test].detach().numpy(), pred_test, average="micro")
    f1_macro_test = f1_score(labels[idx_test].detach().numpy(), pred_test, average="macro")
    # print test set result
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "f1_test= {:.4f} {:.4f}".format(f1_micro_test, f1_macro_test))


if __name__ == '__main__':
    # fix random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    gae_for(args)
