# FastGCN for classification task in cora and citeseer
# Paper: Chen J, Ma T, Xiao C. Fastgcn: fast learning with graph convolutional networks via importance sampling  https://arxiv.org/abs/1801.10247
# test_acc: 0.854 for cora if use training set of 1208; 0.736 if use training set of 140, 
# 0.783 for citeseer if use original segmentation, 0.674 if use training set of 120. 

# Runtime: 2.35s for cora, and 3.48s for citeseer on a single CPU (11th Gen Intel(R) Core(TM) i7-11800H @ 2.30GHZ)
# Cost: N/A
# Description: apply FastGCN to a cora and siteseer, classification
# comment: faster and better then original GCN

import sys
sys.path.append("../src")
sys.path.append("../../../rllm/dataloader")

from utils import sparse_mx_to_torch_sparse_tensor
from utils import get_batches, accuracy
from sampler import Sampler_FastGCN, Sampler_ASGCN
from models import GCN
import argparse
import time
import scipy.sparse as sp
from load_data import load_data

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

st = time.time()


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
    parser.add_argument('--seed', type=int, default=123, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.01,
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


def train(train_ind, train_labels, batch_size, train_times):
    t = time.time()
    model.train()
    for epoch in range(train_times):
        for batch_inds, batch_labels in get_batches(train_ind,
                                                    train_labels,
                                                    batch_size):
            sampled_feats, sampled_adjs, var_loss = model.sampling(
                batch_inds)
            optimizer.zero_grad()
            output = model(sampled_feats, sampled_adjs)
            loss_train = loss_fn(output, batch_labels) + 0.5 * var_loss
            acc_train = accuracy(output, batch_labels)
            loss_train.backward()
            optimizer.step()
    # just return the train loss of the last train epoch
    return loss_train.item(), acc_train.item(), time.time() - t


def test(test_adj, test_feats, test_labels, epoch):
    t = time.time()
    model.eval()
    outputs = model(test_feats, test_adj)
    loss_test = loss_fn(outputs, test_labels)
    acc_test = accuracy(outputs, test_labels)

    return loss_test.item(), acc_test.item(), time.time() - t

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
    mask = np.zeros(lst)
    mask[idx] = 1
    return np.array(mask, dtype=bool)


def nontuple_preprocess_adj(adj):
    adj_normalized = normalize_adj(sp.eye(adj.shape[0]) + adj)
    # adj_normalized = sp.eye(adj.shape[0]) + normalize_adj(adj)
    return adj_normalized.tocsr()


if __name__ == '__main__':
    # load data, set superpara and constant
    args = get_args()
    data, adj, features, labels, idx_train, idx_val, idx_test = load_data(
    args.dataset)
    label_origin = labels.detach().cpu().numpy()

    # move to fit

    train_mask = sample_mask(idx_train, label_origin.shape[0])
    val_mask = sample_mask(idx_val, label_origin.shape[0])
    test_mask = sample_mask(idx_test, label_origin.shape[0])

    # print(f"idx_train.shape: {idx_train.shape}")

    y_train = np.zeros(label_origin.shape)
    y_val = np.zeros(label_origin.shape)
    y_test = np.zeros(label_origin.shape)
    y_train[train_mask, :] = label_origin[train_mask, :]
    y_val[val_mask, :] = label_origin[val_mask, :]
    y_test[test_mask, :] = label_origin[test_mask, :]
    train_index = np.where(train_mask)[0]
    # adj_train = adj[train_index, :][:, train_index]
    y_train = y_train[train_index]
    val_index = np.where(val_mask)[0]
    y_val = y_val[val_index]
    test_index = np.where(test_mask)[0]
    y_test = y_test[test_index]
    # print(f"y_test.shape: {y_test.shape}")
    # print(f"y_val.shape: {y_val.shape}")


    sparse_adj = sp.csr_matrix(adj.to_dense().numpy())
    # norm_adj_train = nontuple_preprocess_adj(idx_train)
    norm_adj = nontuple_preprocess_adj(sparse_adj)
    adj = norm_adj

    # print(f"train_index: {train_index}")
    # print(f"sparse_adj: {sparse_adj}")
    adj_train = sparse_adj[train_index, :][:, train_index]
    norm_adj_train = nontuple_preprocess_adj(adj_train)
    adj_train = norm_adj_train
    train_features = features[train_index]

    layer_sizes = [128, 128]
    input_dim = features.shape[1]
    train_nums = adj_train.shape[0]
    test_gap = args.test_gap
    nclass = y_train.shape[1]

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    # set device
    if args.cuda:
        device = torch.device("cuda")
        print("use cuda")
    else:
        device = torch.device("cpu")

    # data for train and test
    features = torch.FloatTensor(features).to(device)
    train_features = torch.FloatTensor(train_features).to(device)
    y_train = torch.LongTensor(y_train).to(device).max(1)[1]

    test_adj = [adj, adj[test_index, :]]
    test_feats = features
    test_labels = y_test
    test_adj = [sparse_mx_to_torch_sparse_tensor(cur_adj).to(device)
                for cur_adj in test_adj]
    test_labels = torch.LongTensor(test_labels).to(device).max(1)[1]

    # init the sampler
    if args.model == 'Fast':
        sampler = Sampler_FastGCN(None, train_features, adj_train,
                                  input_dim=input_dim,
                                  layer_sizes=layer_sizes,
                                  device=device)
    elif args.model == 'AS':
        sampler = Sampler_ASGCN(None, train_features, adj_train,
                                input_dim=input_dim,
                                layer_sizes=layer_sizes,
                                device=device)
    else:
        print(f"model name error, no model named {args.model}")
        exit()

    # init model, optimizer and loss function
    model = GCN(nfeat=features.shape[1],
                nhid=args.hidden,
                nclass=nclass,
                dropout=args.dropout,
                sampler=sampler).to(device)
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = F.nll_loss
    # loss_fn = torch.nn.CrossEntropyLoss()

    # train and test
    for epochs in range(0, args.epochs // test_gap):
        train_loss, train_acc, train_time = train(np.arange(train_nums),
                                                  y_train,
                                                  args.batchsize,
                                                  test_gap)
        test_loss, test_acc, test_time = test(test_adj,
                                              test_feats,
                                              test_labels,
                                              args.epochs)
        print(f"epchs:{epochs * test_gap}~{(epochs + 1) * test_gap - 1} "
              f"train_loss: {train_loss:.3f}, "
              f"train_acc: {train_acc:.3f}, "
              f"train_times: {train_time:.3f}s "
              f"test_loss: {test_loss:.3f}, "
              f"test_acc: {test_acc:.3f}, "
              f"test_times: {test_time:.3f}s")
    print(f"time: {time.time() - st}")
