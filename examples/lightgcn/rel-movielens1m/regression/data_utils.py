import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import scipy.sparse as sp
import random
import torch
import copy


def create_dataset(args):
    if args.dataset in ['ml-100k', 'ml-1m', 'ml-10m', 'rel-movielens1m/regression/ratings']:
        return ML_Dataset(args)
    elif args.dataset in ['gowalla', 'amazon-book', 'yelp2018']:
        return LightGCN_Dataset(args)
    else:
        raise ValueError('Check args.dataset if right!')


def create_dataloader(dataset, batch_size, training=False):
    if training:
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    else:
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)


class ML_Dataset(object):
    def __init__(self, args):
        self.args = args

        self.dataset = self.args.dataset
        self.data_path = self.args.data_path

        self.train_inter_feat, self.test_inter_feat = self._load_dataframe()  # user-item-interaction DataFrame: user, item, rating, timestamp
        
        self.test_reserved = self.test_inter_feat.copy()

        # user_id, item_id, rating, timestamp
        self.uid_field, self.iid_field, self.rating_field, self.timestamp = self.train_inter_feat.columns
        self.user_num = 6040 + 1
        self.item_num = 3952 + 1
        
        self.train_inter_feat, self.test_inter_feat = self._split_inter_feat()  # DataFrame: user, pos_item_list
        
        # the positive item num and negative item num of each user
        self.train_items_num = [len(i) for i in self.train_inter_feat['MovieID']]
        self.neg_items_num = [self.args.neg_sample_num * len(i) for i in self.train_inter_feat['MovieID']]

    def _load_dataframe(self):
        # '../data/ml-100k/ml-100k.inter'
        train_inter_feat_path = os.path.join(self.data_path + '/' + self.dataset + '/' + 'train.csv')
        test_inter_feat_path = os.path.join(self.data_path + '/' + self.dataset + '/' + 'test.csv')

        if not os.path.isfile(train_inter_feat_path) or not os.path.isfile(test_inter_feat_path):
            raise ValueError(f'File {train_inter_feat_path} or {test_inter_feat_path} not exist.')

        # create DataFrame
        train_inter_feat = pd.read_csv(train_inter_feat_path)
        test_inter_feat = pd.read_csv(test_inter_feat_path)
        
        # decrease 1 for user_id and item_id
        train_inter_feat.iloc[:, 0:2] = train_inter_feat.iloc[:, 0:2].apply(lambda x: x - 1)
        test_inter_feat.iloc[:, 0:2] = test_inter_feat.iloc[:, 0:2].apply(lambda x: x - 1)
         
        return train_inter_feat, test_inter_feat

    def _split_inter_feat(self):
        # modify map
        train_status = self.train_inter_feat.groupby(self.uid_field)[self.iid_field].apply(set).reset_index().rename(
            columns={self.iid_field: 'train_interacted_items'}
        ) 
        test_status = self.test_inter_feat.groupby(self.uid_field)[self.iid_field].apply(set).reset_index().rename(
            columns={self.iid_field: 'test_interacted_items'}
        )  # user-item_dic-interaction DataFrame: user, interacted_items
        
        train_status['MovieID'] = train_status['train_interacted_items'].apply(list)
        test_status['MovieID'] = test_status['test_interacted_items'].apply(list)
        
        train_inter_feat = train_status[[self.uid_field, 'MovieID']]
        test_inter_feat = test_status[[self.uid_field, 'MovieID']]    
        return train_inter_feat, test_inter_feat

    def _sample_negative(self, pos_item, sampling_num):
        neg_item = []
        for i in range(sampling_num):
            while True:
                negitem = random.choice(range(self.item_num))
                if negitem not in pos_item:
                    break
            neg_item.append(negitem)
        return neg_item

    def get_train_dataset(self):
        users, pos_items, neg_items = [], [], []
        mask_index = {}  # dict: used for test
        for row in self.train_inter_feat.itertuples():
            index = getattr(row, 'Index')
            user_id = getattr(row, self.uid_field)
            pos_item = getattr(row, 'MovieID')
            neg_item = self._sample_negative(pos_item, self.neg_items_num[index])

            mask_index[user_id] = pos_item

            users.extend([user_id] * len(neg_item))
            pos_items.extend(pos_item * self.args.neg_sample_num)
            neg_items.extend(neg_item)

        interaction_matrix = self.inter_matrix(users, pos_items)

        train_dataset = TorchDataset(user=torch.LongTensor(users),
                                     pos_item=torch.LongTensor(pos_items),
                                     neg_item=torch.LongTensor(neg_items))
        return train_dataset, interaction_matrix, mask_index

    def get_test_data(self):
        # test_users = list(self.test_inter_feat[self.uid_field])
        # ground_true_items = list(self.test_inter_feat['MovieID'])  # list like [[],[],...,[]] len: n_users
        # return test_users, ground_true_items
        uid_field, iid_field, rating_field, timestamp = self.test_reserved.columns
        print(self.test_reserved)
        User_ID = self.test_reserved[uid_field]
        Movie_ID = self.test_reserved[iid_field]
        Rating = self.test_reserved[rating_field]
        return User_ID, Movie_ID, Rating

    def inter_matrix(self, users, pos_items, form='coo'):
        row = users
        col = pos_items
        data = np.ones(len(row))

        mat = sp.coo_matrix((data, (row, col)), shape=(self.user_num, self.item_num))

        if form == 'coo':
            return mat
        elif form == 'csr':
            return mat.tocsr()
        else:
            raise NotImplementedError(f'Sparse matrix format [{form}] has not been implemented.')


class LightGCN_Dataset(object):
    def __init__(self, args):
        self.args = args

        path = os.path.join(self.args.data_path, self.args.dataset)
        self.train_file = path + '/train.csv'
        self.test_file = path + '/test.csv'

        if self.args.dataset == 'gowalla':
            self.user_num = 29858
            self.item_num = 40981
        elif self.args.dataset == 'amazon-book':
            self.user_num = 52643
            self.item_num = 91599
        elif self.args.dataset == 'yelp2018':
            self.user_num = 31668
            self.item_num = 38048
        elif self.args.dataset == 'rel-movielens1m/regression':
            self.user_num = 6040
            self.item_num = 3952
        else:
            raise ValueError('Check args.dataset if right!')

    def get_train_dataset(self):
        users, pos_items, neg_items = [], [], []
        mask_index = {}
        df = pd.read_csv(self.train_file)
        for index, row in df.iterrows():
            pos_item = [int(row['MovieID'])]
            user_id = int(row['UserID'])
            neg_item = self._sample_negative(pos_item)

            mask_index[user_id] = pos_item

            users.extend([user_id] * len(neg_item))
            pos_items.extend(pos_item * self.args.neg_sample_num)
            neg_items.extend(neg_item)
        interaction_matrix = self._inter_matrix(users, pos_items)

        train_dataset = TorchDataset(user=torch.LongTensor(users),
                                     pos_item=torch.LongTensor(pos_items),
                                     neg_item=torch.LongTensor(neg_items))
        return train_dataset, interaction_matrix, mask_index

    def _sample_negative(self, pos_item):
        sampling_num = len(pos_item) * self.args.neg_sample_num
        neg_item = []
        for i in range(sampling_num):
            while True:
                negitem = random.choice(range(self.item_num))
                if negitem not in pos_item:
                    break
            neg_item.append(negitem)
        return neg_item

    def _inter_matrix(self, users, pos_items, form='coo'):
        row = users
        col = pos_items
        data = np.ones(len(row))

        mat = sp.coo_matrix((data, (row, col)), shape=(self.user_num, self.item_num))

        if form == 'coo':
            return mat
        elif form == 'csr':
            return mat.tocsr()
        else:
            raise NotImplementedError(f'Sparse matrix format [{form}] has not been implemented.')

    def get_test_data(self):
        test_users = list(range(self.user_num))

        ground_true_items = []  # list like [[],[],...,[]] len: n_users
        df = pd.read_csv(self.test_file)
        for index, row in df.iterrows():
            items = [int(row['MovieID'])]
            ground_true_items.append(items)
        return test_users, ground_true_items


class TorchDataset(Dataset):
    def __init__(self, user, pos_item, neg_item):
        super(Dataset, self).__init__()

        self.user = user
        self.pos_item = pos_item
        self.neg_item = neg_item

    def __len__(self):
        return len(self.user)

    def __getitem__(self, idx):
        return self.user[idx], self.pos_item[idx], self.neg_item[idx]

