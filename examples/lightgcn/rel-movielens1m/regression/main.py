import argparse
import numpy as np
import time

from data_utils import create_dataset, create_dataloader
from utils import init_seed
from model import LightGCN
from trainer import Trainer


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=2020, help="the random seed")
parser.add_argument('--data_path', type=str, default='../data', help="data path")
parser.add_argument('--dataset', type=str, default='rel-movielens1m/regression/ratings', help="dataset of ratings")
parser.add_argument('--epochs', type=int, default=3, help="")
parser.add_argument('--batch_size', type=int, default=2048, help="")
parser.add_argument('--test_batch_size', type=int, default=2048, help='')
parser.add_argument('--learning_rate', type=float, default=1e-2, help="")
parser.add_argument('--reg_weight', type=float, default=1e-2, help="the weight decay of BPR Loss")
parser.add_argument('--weight_decay', type=float, default=0, help="the weight decay of optimizer")
parser.add_argument('--topk', type=int, default=20, help="")
parser.add_argument('--metrics', type=list, default=['Recall', 'MRR', 'NDCG', 'Hit', 'Precision'], help="")
parser.add_argument('--optimizer', type=str, default='adam',
                    help='the optimizer to update parameters. optimizer in [adam, sgd, adagrad, rmsprop]')
parser.add_argument('--data_split_ratio', type=list, default=[0.8],
                    help="the ration to split dataset to train, eval and test")
parser.add_argument('--neg_sample_num', type=int, default=1, help="each user-item interaction sampled negative num")
parser.add_argument('--embedding_size', type=int, default=64, help="the latent vector embedding size")
parser.add_argument('--n_layers', type=int, default=2, help="the graph convolution layer num")
parser.add_argument('--device', type=str, default='cuda', help="")
parser.add_argument('--gpu_id', type=int, default=0, help="")
parser.add_argument('--neg_prefix', type=str, default='neg_', help="")
parser.add_argument('--tensorboard_dir', type=str, default='./tensorboard_log', help='the path to save tensorboard')

args = parser.parse_args()

# init random seed
init_seed(args.seed)

print('Used dataset is {}'.format(args.dataset))
# load dataset
dataset = create_dataset(args)
# split dataset
train_dataset, interaction_matrix, mask_index = dataset.get_train_dataset()
test_users, test_movies, test_scores= dataset.get_test_data()

# load dataloader
train_data = create_dataloader(train_dataset, args.batch_size, training=True)
test_data = create_dataloader(test_users, args.test_batch_size, training=False)

# get the model
model = LightGCN(args, dataset, interaction_matrix).to(args.device)

print("----------Training-----------------------")
trainer = Trainer(args, model)
for epoch in range(args.epochs):
    # training
    trainer.train_an_epoch(train_data, epoch_id=epoch + 1)
    # print(test_users.shape)
    # print(test_movies.shape)
    # print(test_scores.shape)
    # print(ground_true_items)
    # testing
    trainer.evaluate(test_users, test_movies, test_scores)
    # trainer.evaluate_regression(test_data, ground_true_items, mask_index, epoch_id=epoch + 1)
