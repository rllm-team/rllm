# KGAT
# KGAT: Knowledge Graph Attention Network for Recommendation
# https://arxiv.org/abs/1905.07854
# F1-Ma: 0.1542642875096072 F1-Mi: 0.16810245995610731
# about 690s on Tesla V100-SXM2
# N/A
# python main_kgat.py
import sys
import random
from time import time
import torch
import torch.optim as optim
import numpy as np

sys.path.append("../../../kgat")
from model.KGAT import MLP_KGAT
from parsers.parser_kgat import parse_kgat_args_clf
from utils.log_helper import create_log_id, logging_config
from utils.metrics import calc_mse
from utils.model_helper import load_model
from utils.io_helper import read_interactions
import logging
from data_loader.movie_loader_kgat import DataLoaderKGAT
sys.path.append("../../../../rllm/dataloader")
from load_data import load_data

# Load data
data, adj, features, labels, idx_train, idx_val, idx_test = \
    load_data('movielens-classification')

args = parse_kgat_args_clf()
# seed
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

log_save_id = create_log_id(args.save_dir)
logging_config(
    folder=args.save_dir,
    name='log{:d}'.format(log_save_id),
    no_console=False)
logging.info(args)

# GPU / CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load data
data = DataLoaderKGAT(args, logging)

val_rating = read_interactions(
    "../datasets/rel-movielens/val_rating.txt", float
    )
val_interaction = read_interactions("../datasets/rel-movielens/val.txt", int)
test_rating = read_interactions(
    "../datasets/rel-movielens/test_rating.txt",
    float)
test_interaction = read_interactions(
    "../datasets/rel-movielens/test.txt",
    int)
test_users = (torch.Tensor(list(test_rating.keys()))
              .to(device) + data.n_entities)
test_users = test_users.long()
interactions = []
rating_true = []
for user in test_users:
    key = user.item() - data.n_entities
    interactions.extend(test_interaction[key].tolist())
    rating_true.extend(test_rating[key].tolist())
interactions = torch.Tensor(interactions).long().to(device)
rating_true = torch.Tensor(rating_true).to(device)


def evaluate(model, device):
    model.eval()
    score = model.calc_score(test_users.long(), interactions)
    score_list = []
    for i, user in enumerate(test_users):
        key = user.item() - data.n_entities
        curr_score = score[i, test_interaction[key].to(device)]
        score_list.extend(curr_score.tolist())
    score_list = torch.Tensor(score_list).to(device)
    return calc_mse(score_list, rating_true).item()


def mse_loss(model, batch_user, batch_item, true_relation):
    score = model.calc_score(batch_user.long(), batch_item.long())
    score = torch.diag(score).to(device)
    return calc_mse(score, true_relation.float())


def train(args):
    if args.use_pretrain == 1:
        user_pre_embed = torch.tensor(data.user_pre_embed)
        item_pre_embed = torch.tensor(data.item_pre_embed)
    else:
        user_pre_embed, item_pre_embed = None, None

    # construct model & optimizer
    model = MLP_KGAT(
        args,
        data.n_users,
        data.n_entities,
        data.n_relations,
        data.A_in,
        user_pre_embed,
        item_pre_embed
    )
    if args.use_pretrain == 2:
        model = load_model(model, args.pretrain_model_path)

    model.to(device)

    cf_optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # kg_optimizer = optim.Adam(model.parameters(), lr=args.lr)
    classifier_optimizer = optim.Adam(model.parameters(), lr=args.lr)

    total_time = 0.

    # train model
    for epoch in range(1, args.n_epoch + 1):
        logging.info(f"MSE: {evaluate(model, device)}")
        epoch_start = time()
        model.train()

        # train cf

        cf_total_loss = 0
        n_cf_batch = data.n_cf_train // data.cf_batch_size + 1

        for iter in range(1, n_cf_batch + 1):
            time2 = time()
            cf_batch_user, cf_batch_relation, \
                cf_batch_pos_item, cf_batch_neg_item = \
                data.generate_kg_batch(
                    data.train_kg_dict,
                    data.cf_batch_size,
                    data.n_users_entities)
            cf_batch_relation[
                cf_batch_relation > data.type_of_scores] = \
                cf_batch_relation[
                    cf_batch_relation > data.type_of_scores
                ] - data.type_of_scores
            cf_batch_relation = cf_batch_relation.to(device)
            cf_batch_user = cf_batch_user.to(device)
            cf_batch_pos_item = cf_batch_pos_item.to(device)
            cf_batch_neg_item = cf_batch_neg_item.to(device)

            cf_batch_loss = model(
                cf_batch_user, cf_batch_pos_item, cf_batch_neg_item,
                mode='train_cf')
            mse = mse_loss(
                model,
                cf_batch_user,
                cf_batch_pos_item,
                cf_batch_relation)
            cf_batch_loss = mse

            if np.isnan(cf_batch_loss.cpu().detach().numpy()):
                logging.info(
                    'ERROR (CF Training): \
                        Epoch {:04d} Iter {:04d} / {:04d} Loss is nan.'
                    .format(epoch, iter, n_cf_batch)
                )
                sys.exit()

            cf_batch_loss.backward()
            cf_optimizer.step()
            cf_optimizer.zero_grad()
            cf_total_loss += cf_batch_loss.item()

            if (iter % args.cf_print_every) == 0:
                logging.info(
                    'CF Training: Epoch {:04d} Iter {:04d} / {:04d} | \
                        loss {:04f} \
                        Time {:.1f}s'
                    .format(
                        epoch, iter, n_cf_batch, cf_batch_loss.item(),
                        time() - time2)
                    )
        total_time += time() - epoch_start

    model.train_classifier(
        labels.to(device),
        idx_train.to(device),
        idx_test.to(device),
        classifier_optimizer)
    print(f"Total training time: {total_time} s")
    torch.save(model, 'trained_model/KGAT/rel-movielens/kgat.pth')


if __name__ == '__main__':
    train(args)
    # predict(args)
