# GraphSage
# Inductive Representation Learning on Large Graphs
# https://arxiv.org/abs/1706.02216
# MSE: 1.3283003568649292
# 3.6268s
# N/A
# python train.py
import sys
sys.path.append("../../../../rllm/dataloader")
sys.path.append("../../../graphsage")

import time
import argparse
import numpy as np

# import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# from utils import load_data

from load_data import load_data
from models import GraphSage
from utils import adj_matrix_to_list, multihop_sampling

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
parser.add_argument('--lr', type=float, default=0.005,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=list, default=[64],
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
data, adj, features, labels, idx_train, idx_val, idx_test = \
    load_data('movielens-regression')
# labels = labels.argmax(dim=-1)

node_index = torch.arange(0, adj.shape[0])
node_index_user = torch.arange(0, adj.indices()[0, -1]+1)
node_index_movie = torch.arange(adj.indices()[1, 0], adj.indices()[1, -1]+1)
node_index_train_user = \
    torch.unique(adj.indices()[:, idx_train][0], return_counts=False)
node_index_train_movie = \
    torch.unique(adj.indices()[:, idx_train][1], return_counts=False)
node_index_test_user = \
    torch.unique(adj.indices()[:, idx_test][0], return_counts=False)
node_index_test_movie = \
    torch.unique(adj.indices()[:, idx_test][1], return_counts=False)

label_mat = torch.sparse_coo_tensor(
    adj.indices(),
    labels,
    size=adj.shape,
    requires_grad=False).float()
label_mat_test = torch.sparse_coo_tensor(
    adj.indices()[:, idx_test],
    labels[idx_test],
    size=adj.shape,
    requires_grad=False).float()
test_adjacency_dict, test_label_dict = \
    adj_matrix_to_list(
        label_mat_test,
        node_index_test_movie,
        label_mat_test,
        "movie-reg")
adjacency_dict, label_dict = \
    adj_matrix_to_list(adj, node_index_movie, label_mat, "movie-reg")
# print(adj)

# Model and optimizer

NUM_NEIGHBORS_LIST = [25]
NUM_BATCH_PER_EPOCH = 5
batch_size = 16

model = GraphSage(input_dim=features.shape[1], hidden_dim=args.hidden,
                  num_neighbors_list=NUM_NEIGHBORS_LIST)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)
loss_func = nn.MSELoss()
# loss_func = F.cross_entropy
DEVICE = "cpu"

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    node_index = node_index.cuda()
    node_index_movie = node_index_movie.cuda()
    node_index_user = node_index_user.cuda()
    node_index_train_user = node_index_train_user.cuda()
    node_index_train_movie = node_index_train_movie.cuda()
    node_index_test_user = node_index_test_user.cuda()
    node_index_test_movie = node_index_test_movie.cuda()


def train(epoch):
    t = time.time()
    model.train()
    loss_lst = []
    # if epoch % 5 == 0: test()
    for batch in range(NUM_BATCH_PER_EPOCH):
        optimizer.zero_grad()
        # sample from node_index_movie first
        batch_node_movie_train = node_index_movie[
            torch.randint(0, len(node_index_movie), (batch_size,))]
        batch_sampling_result, batch_sampling_label = \
            multihop_sampling(
                batch_node_movie_train,
                NUM_NEIGHBORS_LIST,
                adjacency_dict,
                "movie-reg",
                label_dict
                )
        batch_sampling_x = [
            features[idx].float().to(DEVICE) for idx in batch_sampling_result]
        output = model(batch_sampling_x)
        # batch_sampling_label = \
        #     F.one_hot(batch_sampling_label.long()-1, 5).float().to(DEVICE)
        loss_train = loss_func(output, batch_sampling_label.float().cuda())
        loss_lst.append(loss_train.detach().item())
        loss_train.backward()
        optimizer.step()

    print(
        'Epoch: {:04d}'.format(epoch+1),
        'loss_train: {:.4f}'.format(sum(loss_lst)/len(loss_lst)),
        'time: {:.4f}s'.format(time.time() - t))


def test():
    model.eval()
    with torch.no_grad():
        batch_sampling_result, batch_sampling_label = \
            multihop_sampling(
                node_index_test_movie,
                NUM_NEIGHBORS_LIST,
                adjacency_dict,
                "movie-reg",
                label_dict)
        batch_sampling_x = [
            features[idx].float().to(DEVICE) for idx in batch_sampling_result]
        pred = model.test(
            batch_sampling_x,
            node_index_test_movie,
            test_adjacency_dict,
            features)
        pred_list = []
        label_list = []
        for key, value in pred.items():
            pred_list.extend(value.tolist())
            label_list.extend(test_label_dict[key].tolist())
        pred_list = torch.tensor(pred_list).squeeze()
        label_list = torch.tensor(label_list)
        print("test MSE:", F.mse_loss(pred_list, label_list).item())


# test()
# Train model
t_total = time.time()
for epoch in range(args.epochs):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()
