# GraphSage
# Inductive Representation Learning on Large Graphs
# https://arxiv.org/abs/1706.02216
# micro: 0.34596944320143197; macro: 0.11430499992755531
# 6.09s
# N/A
# python train.py

import sys
import time
import argparse
import numpy as np

# import math
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score

# from utils import load_data
sys.path.append("../../../../rllm/dataloader")
sys.path.append("../../../graphsage")
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
parser.add_argument('--epochs', type=int, default=50,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.005,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=list, default=[64, 18],
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

_device = 'cpu'
if args.cuda:
    _device = 'cuda'
# Load data
data, adj, features, labels, idx_train, idx_val, idx_test = \
    load_data('movielens-classification', device=_device)
# labels = labels.argmax(dim=-1)


adjacency_dict = adj_matrix_to_list(adj)
# print(adj)

# Model and optimizer

NUM_NEIGHBORS_LIST = [25, 10]
NUM_BATCH_PER_EPOCH = 5
batch_size = 64
model = GraphSage(input_dim=features.shape[1], hidden_dim=args.hidden,
                  num_neighbors_list=NUM_NEIGHBORS_LIST)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)
# loss_func = nn.BCEWithLogitsLoss()
loss_func = F.cross_entropy
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


def train(epoch):
    t = time.time()
    model.train()
    loss_lst = []
    for batch in range(NUM_BATCH_PER_EPOCH):
        optimizer.zero_grad()
        rand_idx = torch.randint(0, len(idx_train), (batch_size,))
        batch_src_index = idx_train[rand_idx]
        batch_src_label = labels[batch_src_index].float().to(DEVICE)
        batch_sampling_result = multihop_sampling(
            batch_src_index,
            NUM_NEIGHBORS_LIST,
            adjacency_dict,
            "movie-cla"
            )
        batch_sampling_x = [
            features[idx].float().to(DEVICE) for idx in batch_sampling_result
            ]
        output = model(batch_sampling_x)
        loss_train = loss_func(torch.sigmoid(output), batch_src_label)
        loss_lst.append(loss_train.detach().item())
        loss_train.backward()
        optimizer.step()
    # if not args.fastmode:
    #     # Evaluate validation set performance separately,
    #     # deactivates dropout during validation run.
    #     model.eval()
    #     output = model(features, adj)

    print(
        'Epoch: {:04d}'.format(epoch+1),
        'loss_train: {:.4f}'.format(sum(loss_lst)/len(loss_lst)),
        'time: {:.4f}s'.format(time.time() - t)
        )
    # print('acc_train: {:.4f}'.format(acc_train.item()/num))


def test():
    model.eval()
    with torch.no_grad():
        test_sampling_result = multihop_sampling(
            idx_test,
            NUM_NEIGHBORS_LIST,
            adjacency_dict,
            "movie-cla"
            )
        test_x = [
            features[idx].float().to(DEVICE) for idx in test_sampling_result
            ]
        test_logits = model(test_x)
        test_logits = torch.sigmoid(test_logits).cpu()
        pred = np.where(test_logits > 0.5, 1, 0)
        test_label = labels[idx_test].float().cpu()
        f1_micro_test = f1_score(test_label, pred, average="micro")
        f1_macro_test = f1_score(test_label, pred, average="macro")
        print(f"micro: {f1_micro_test}; macro: {f1_macro_test}")


# Train model
t_total = time.time()
for epoch in range(args.epochs):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()
