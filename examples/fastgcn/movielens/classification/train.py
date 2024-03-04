# FastGCN for classification task in rel-movielens1M
# Paper: Chen J, Ma T, Xiao C. Fastgcn: fast learning with graph convolutional networks via importance sampling  https://arxiv.org/abs/1801.10247
# f1_micro_test: 0.346 f1_macro_test: 0.121
# Runtime: 12.778s on a single CPU (11th Gen Intel(R) Core(TM) i7-11800H @ 2.30GHZ)
# Cost: N/A
# Description: apply FastGCN to a movielens, classification
# comment: faster and better then original GCN


from __future__ import division
from __future__ import print_function
import sys
sys.path.append("../../src")
sys.path.append("../../../../rllm/dataloader")
import numpy as np
import time
import argparse
from load_data import load_data
import torch
import torch.optim as optim
import torch.nn as nn
from models import GCN
from sampler import Sampler_FastGCN, Sampler_ASGCN
from utils_movielens import get_batches
from utils_movielens import sparse_mx_to_torch_sparse_tensor
from sklearn.metrics import f1_score
import scipy.sparse as sp
import warnings
warnings.filterwarnings("ignore")

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
    parser.add_argument('--seed', type=int, default=5, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=100,
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
    mask = np.zeros(lst)
    mask[idx] = 1
    return np.array(mask, dtype=bool)


def nontuple_preprocess_adj(adj):
    adj_normalized = normalize_adj(sp.eye(adj.shape[0]) + adj)
    # adj_normalized = sp.eye(adj.shape[0]) + normalize_adj(adj)
    return adj_normalized.tocsr()


# load data, set superpara and constant
args = get_args()
data, adj, features, labels, idx_train, idx_val, idx_test = load_data(
    'movielens-classification')
# print(f"features: {features}")
# print(f"adj: {adj}")
# print(f"idx_test.shape: {idx_test.shape}")
# print(f"labels.shape: {labels.shape}")
# print(f"idx_train.shape: {idx_train.shape}")
# print(f"idx_val.shape: {idx_val.shape}")
# print(f"idx_test.shape: {idx_test.shape}")
# print(f"labels.shape: {labels.shape}")
label_origin = labels.detach().cpu().numpy()
# print(f"label_origin: {label_origin}")
# print(data)

# labels = labels.argmax(dim=-1)

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


sparse_adj = sp.csr_matrix(adj)
# norm_adj_train = nontuple_preprocess_adj(idx_train)
norm_adj = nontuple_preprocess_adj(sparse_adj)
adj = norm_adj

# print(f"train_index: {train_index}")
# print(f"sparse_adj: {sparse_adj}")
adj_train = sparse_adj[train_index, :][:, train_index]
norm_adj_train = nontuple_preprocess_adj(adj_train)
adj_train = norm_adj_train
train_features = features[train_index]

# print(f"adj: {adj}")
# print(f"features: {features}")
# print(f"adj_train: {adj_train}")
# print(f"train_features: {train_features}")
# print(f"y_train: {y_train}")
# print(f"y_test: {y_test}")
# print(f"test_index: {idx_test}")

# adj, features, adj_train, train_features, y_train, y_test, test_index = \
# load_data(args.dataset)

layer_sizes = [128, 128]
input_dim = features.shape[1]
train_nums = adj_train.shape[0]
test_gap = args.test_gap
nclass = y_train.shape[1]
# print(f"n_class: {nclass}")

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
# print(f"y_train: {y_train}")
# y_train = torch.LongTensor(y_train).to(device).max(1)[1]
# print(f"y_train: {y_train}")
'''这一步把y_train 从one-hot变成一个数字了'''

# test_adj = [adj, adj[test_index, :]]
test_adj = [adj, adj[idx_test, :]]
test_feats = features
test_labels = y_test
test_adj = [sparse_mx_to_torch_sparse_tensor(cur_adj).to(device)
            for cur_adj in test_adj]

val_adj = [adj, adj[idx_val, :]]
val_feats = features
val_labels = y_val
val_adj = [sparse_mx_to_torch_sparse_tensor(cur_adj).to(device)
           for cur_adj in val_adj]

# test_labels = torch.LongTensor(test_labels).to(device).max(1)[1]
# print(f"test_labels: {test_labels}")

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
# loss_fn = F.nll_loss
# loss_fn = nn.MSELoss()
loss_fn = nn.BCEWithLogitsLoss()
# loss_fn = torch.nn.CrossEntropyLoss()


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
            sampled_feats = torch.tensor(sampled_feats, requires_grad=True)
            # print(f"sampled_adjs: {sampled_adjs}")
            adj_list = []
            for adj_item in sampled_adjs:
                adj_in = adj_item.to_dense().to(device)
                adj_in = torch.tensor(adj_in, requires_grad=True)
                # print(f"adj_in: {adj_in}")
                adj_list.append(adj_in)
            output = model(sampled_feats, adj_list)
            # # print(f"output: {output}")
            batch_labels = torch.tensor(
                batch_labels, requires_grad=True, dtype=torch.float).to(device)

            pred = np.where(output > -1, 1, 0)

            loss_train = loss_fn(
                output, batch_labels)

            loss_train = loss_train.float().to(device)
            # 111

            f1_micro_train = f1_score(
                pred, batch_labels.detach(), average="micro")
            f1_macro_train = f1_score(
                pred, batch_labels.detach(), average="macro")

            # print(f"output: {output.shape}")
            # print(f"batch_labels: {batch_labels.shape}")

            # acc_train = accuracy(output, batch_labels)
            loss_train.backward()

            optimizer.step()
    # just return the train loss of the last train epoch
    return loss_train.item(), time.time() - t, f1_micro_train, f1_macro_train


def test(test_adj, test_feats, test_labels, epoch):
    t = time.time()
    model.eval()
    outputs = model(test_feats, test_adj)
    outputs = torch.tensor(outputs, requires_grad=True).to(device)
    test_labels = torch.tensor(
        test_labels, requires_grad=True).to(device)
    pred = np.where(outputs > -1, 1, 0)

    # counts = 0
    # for i in range(test_labels.shape[0]):
    #     for j in range(test_labels.shape[1]):
    #         num = test_labels[i][j]
    #         if (num == 1):
    #             counts += 1
    # print(f"counts_1: {counts}")
    # print(f"outputs: {outputs}")
    # print(f"test_labels: {test_labels.shape}")
    loss_test = loss_fn(outputs, test_labels)
    f1_micro_test = f1_score(pred, test_labels.detach(), average="micro")
    f1_macro_test = f1_score(pred, test_labels.detach(), average="macro")
    # acc_test = accuracy(outputs, test_labels)

    return loss_test.item(), time.time() - t, f1_micro_test, f1_macro_test


if __name__ == '__main__':

    # train and test
    for epochs in range(0, args.epochs // test_gap):

        (train_loss,
         train_time,
         f1_micro_train,
         f1_macro_train) = train(np.arange(train_nums),
                                 y_train,
                                 args.batchsize,
                                 test_gap)

        val_loss, val_time, f1_micro_val, f1_macro_val = test(val_adj,
                                                              val_feats,
                                                              val_labels,
                                                              args.epochs)
        print(f"epchs:{epochs * test_gap}~{(epochs + 1) * test_gap - 1} "
              f"train_loss: {train_loss:.3f}, "
              #   f"train_acc: {train_acc:.3f}, "
              f"train_times: {train_time:.3f}s "
              f"f1_micro_train: {f1_micro_train:.3f} "
              f"f1_macro_train: {f1_macro_train:.3f} "
              f"val_times: {val_time:.3f}s "
              f"f1_micro_val: {f1_micro_val:.3f} "
              f"f1_macro_val: {f1_macro_val:.3f} "
              )
        #   f"test_loss: {test_loss:.3f}, "
        #   f"test_acc: {test_acc:.3f}, "
        #   f"test_times: {test_time:.3f}s")
    test_loss, test_time, f1_micro_test, f1_macro_test = test(test_adj,
                                                              test_feats,
                                                              test_labels,
                                                              args.epochs)
    print(f"test_times: {test_time:.3f}s "
          f"f1_micro_test: {f1_micro_test:.3f} "
          f"f1_macro_test: {f1_macro_test:.3f} "
          )

    print(time.time() - st)
