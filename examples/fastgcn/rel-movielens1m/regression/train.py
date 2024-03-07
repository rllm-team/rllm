# FastGCN for regression task in rel-movielens1M
# Paper: Chen J, Ma T, Xiao C. Fastgcn: fast learning with graph convolutional
# networks via importance sampling  https://arxiv.org/abs/1801.10247
# Test MSE Loss: 1.346
# Runtime: 20.015s on a single CPU (11th Gen Intel(R) Core(TM) i7-11800H @ 2.30GHZ)
# Cost: N/A
# Description: apply FastGCN to a small set of movielens, add downsample function and adjust the sampling class to avoid high memory and cpu occupation

# Comment: We've only implemented FastGCN to a small set of movielens in this regression problem. Running regression on the whole movielens dataset would lead to OOM error. (We've try it on a computer with CPU (11th Gen Intel(R) Core(TM) i7-11800H @ 2.30GHZ) and RAM 32GB).
# Unlike other GCN improvements that only adjust the model layer or use a simpler sampling method, FastGCN samples the graph for every layer of the model. Nodes are sampled one by one with a probability proportional to the 2-norm of its column in the matrix to minimize some variants. And unlike classification problem like cora, pubmed and , whose features and labels are all on nodes, the regression problem focus on the features on the edges. If we do the sampling in every training batch, then we're actually dealing with a training dataset of about 7M edges and 10K nodes for [every batch and every layer]. What's worse, the original code used alot of numpy matrix to finish the sampling, and performed type conversion between [Torch.spaseTensor, Torch.sparse_coo_tensor, Torch.denseTensor, scipy.sparse_mx, and numpy.array], which is also a reason for the high memory and cpu occupation.
# in all, if you'd like to do regression with FastGCN, plz wait for further optimization.

from __future__ import division
from __future__ import print_function
import sys
sys.path.append("../../src")
sys.path.append("../../../../rllm/dataloader")


import random
import numpy as np
import time
import argparse
from load_data import load_data
import torch
import torch.optim as optim
import torch.nn as nn
from models import Model
from sampler import Sampler_FastGCN
from utils_movielens import get_batches, sample_more, drop_adj
import scipy.sparse as sp
import warnings


warnings.filterwarnings("ignore")

time_st = time.time()


def get_args():
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cora',
                        help='dataset name.')
    # model can be "Fast" or "AS"
    parser.add_argument('--model', type=str, default='Fast',
                        help='model name.')
    parser.add_argument('--test_gap', type=int, default=10,
                        help='the train epochs between two test')
    parser.add_argument('--no-cuda', action='store_true', default=True,
                        help='Disables CUDA training.')
    parser.add_argument('--fastmode', action='store_true', default=False,
                        help='Validate during training pass.')
    parser.add_argument('--seed', type=int, default=5, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=16,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--batchsize', type=int, default=256,
                        help='batchsize for train')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    return args


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def sample_mask(idx, lst):
    """Create mask."""
    mask = torch.zeros(lst)
    mask[idx] = 1
    return torch.tensor(mask, dtype=bool).to(device)


def nontuple_preprocess_adj(adj):
    adj_normalized = normalize_adj(sp.eye(adj.shape[0]) + adj)
    # adj_normalized = sp.eye(adj.shape[0]) + normalize_adj(adj)
    return adj_normalized.tocsr()


# load data, set superpara and constant
args = get_args()
# set device
if args.cuda:
    device = torch.device("cuda")
    print("use cuda")
else:
    device = torch.device("cpu")
data, adj, features, labels, idx_train, idx_val, idx_test = load_data(
    'movielens-regression')

# 992524 edges 9923 nodes
N = 992524
S = 500
T = 2400


# move to fit
idx_val = torch.LongTensor(random.sample(range(N), S))
adj_val, features_val, labels_val = sample_more(adj, features, labels, idx_val)

idx_test = torch.LongTensor(random.sample(range(N), S))
adj_test, features_test, labels_test = sample_more(
    adj, features, labels, idx_test)

idx_train = torch.LongTensor(random.sample(range(N), T))
adj_train, features_train, labels_train = sample_more(
    adj, features, labels, idx_train)


layer_sizes = [128, 128]
input_dim = features.shape[1]
# train_nums = adj_train.shape[0]
train_nums = adj_train._nnz()
val_nums = adj_val._nnz()
test_nums = adj_test._nnz()
test_gap = args.test_gap


dense_adj_train = adj_train.to_dense().numpy()
sparse_adj_train = sp.coo_matrix(dense_adj_train)
norm_adj_train = nontuple_preprocess_adj(sparse_adj_train)
adj_train = norm_adj_train

dense_adj_val = adj_val.to_dense().numpy()
sparse_adj_val = sp.coo_matrix(dense_adj_val)
norm_adj_val = nontuple_preprocess_adj(sparse_adj_val)
adj_val = norm_adj_val


dense_adj_test = adj_test.to_dense().numpy()
sparse_adj_test = sp.coo_matrix(dense_adj_test)
norm_adj_test = nontuple_preprocess_adj(sparse_adj_test)
adj_test = norm_adj_test

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


# # data for train and test
features = torch.FloatTensor(features).to(device)
features_train = torch.FloatTensor(features_train).to(device)


# init the sampler
if args.model == 'Fast':
    sampler = Sampler_FastGCN(None, features, adj_train,
                              input_dim=input_dim,
                              layer_sizes=layer_sizes,
                              device=device)

    sampler_val = Sampler_FastGCN(None, features, adj_val,
                                  input_dim=input_dim,
                                  layer_sizes=layer_sizes,
                                  device=device)

    sampler_test = Sampler_FastGCN(None, features, adj_test,
                                   input_dim=input_dim,
                                   layer_sizes=layer_sizes,
                                   device=device)

else:
    print(f"model name error, no model named {args.model}")
    exit()


# init model, optimizer and loss function
model = Model(nfeat=features.shape[1],
              nhid=args.hidden,
              dropout=args.dropout,
              sampler=sampler).to(device)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)
loss_fn = nn.MSELoss()


def train(train_ind, train_labels, batch_size, train_times):
    t = time.time()
    model.train()

    for epoch in range(train_times):

        for batch_inds, batch_labels in get_batches(train_ind,
                                                    train_labels,
                                                    batch_size):
            sampled_feats, sampled_adjs, var_loss = model.sampling(
                batch_inds)

            adj_list = []
            adj_coo_list = []
            for adj_item in sampled_adjs:
                adj_coo_list.append(adj_item)
                adj_in = adj_item.to_dense().to(device)
                adj_in = torch.tensor(adj_in, requires_grad=True)
                adj_list.append(adj_in)

            sampled_feats = torch.tensor(sampled_feats, requires_grad=True)

            adj_drop_list = []
            for adj_coo in adj_coo_list:
                adj_drop_list.append(drop_adj(adj_coo, device))

            optimizer.zero_grad()

            output = model(sampled_feats,  adj_coo_list[1], adj_drop_list)

            batch_labels = torch.tensor(
                batch_labels, requires_grad=True, dtype=torch.float).to(device)

            loss_train = loss_fn(
                output, batch_labels)

            loss_train = loss_train.float().to(device)
            loss_train.backward()
            optimizer.step()
    # just return the train loss of the last train epoch
    return loss_train.item(), time.time() - t


def test(test_inds, test_labels, epoch, sampler):
    t = time.time()
    model.eval()

    test_feats, test_adjs, var_loss = sampler.sampling(
        test_inds)

    adj_list = []
    adj_coo_list = []
    for adj_item in test_adjs:
        adj_coo_list.append(adj_item)
        adj_in = adj_item.to_dense().to(device)
        adj_in = torch.tensor(adj_in, requires_grad=True)
        adj_list.append(adj_in)

    test_feats = torch.tensor(test_feats, requires_grad=True)

    adj_drop_list = []
    for adj_coo in adj_coo_list:
        adj_drop_list.append(drop_adj(adj_coo, device))

    optimizer.zero_grad()

    output = model(test_feats,  adj_coo_list[1], adj_drop_list)

    loss_test = loss_fn(
        output, test_labels)

    return loss_test.item(), time.time() - t


if __name__ == '__main__':

    # train and test
    for epochs in range(0, args.epochs // test_gap):
        train_loss, train_time = train(np.arange(train_nums),
                                       labels_train,
                                       args.batchsize,
                                       test_gap)

        val_loss, val_time = test(np.arange(val_nums),
                                  labels_val,
                                  args.epochs,
                                  sampler_val)
        print(f"epchs:{epochs * test_gap}~{(epochs + 1) * test_gap - 1} "
              f"train_loss: {train_loss:.3f}, "
              #   f"train_acc: {train_acc:.3f}, "
              f"train_times: {train_time:.3f}s "
              f"val_times: {val_time:.3f}s "
              )
    test_loss, test_time = test(np.arange(test_nums),
                                labels_test,
                                args.epochs,
                                sampler_test)
    print(f"test_times: {test_time:.3f}s "
          f"test_loss: {test_loss:.3f} "
          )

    time_end = time.time()
    print(f"time: {time_end - time_st}")
