import os
import collections

import torch
import numpy as np
import pandas as pd
import scipy.sparse as sp

from data_loader.loader_base import DataLoaderBase


class DataLoaderKGAT(DataLoaderBase):

    def __init__(self, args, logging):
        super().__init__(args, logging)

        self.cf_batch_size = args.cf_batch_size
        self.kg_batch_size = args.kg_batch_size
        self.test_batch_size = args.test_batch_size

        cache_path = "../datasets/rel-movielens/cache_dict.pkl"
        if not os.path.exists(cache_path):
            kg_data = self.load_kg(self.kg_file)
            self.type_of_scores = max(kg_data['r'])
            self.construct_data(kg_data)
            self.print_info(logging)
            self.laplacian_type = args.laplacian_type
            self.create_adjacency_dict()
            self.create_laplacian_dict()
            self.store_cache(cache_path)
        else:
            self.load_cache(cache_path)
            self.print_info(logging)

    def statistic_cf(self):
        self.val_file = os.path.join(self.data_dir, 'val.txt')
        self.cf_val_data, self.val_user_dict = self.load_cf(self.val_file)
        self.n_users = max(
            max(self.cf_train_data[0]),
            max(self.cf_test_data[0]),
            max(self.cf_val_data[0])) + 1
        self.n_items = max(
            max(self.cf_train_data[1]),
            max(self.cf_test_data[1]),
            max(self.cf_val_data[1])) + 1
        self.n_cf_train = len(self.cf_train_data[0])
        self.n_cf_test = len(self.cf_test_data[0])
        self.n_cf_val = len(self.cf_val_data[0])

    def construct_data(self, kg_data):
        # add inverse kg data
        self.n_entities = max(kg_data['t'])+1
        # inverse_kg_data = kg_data.copy()
        # inverse_kg_data = inverse_kg_data.rename(
        # {'h': 't', 't': 'h'}, axis='columns')
        # inverse_kg_data['r'] += n_relations
        # kg_data = pd.concat([kg_data, inverse_kg_data],
        # axis=0, ignore_index=True, sort=False)

        # re-map user id
        # kg_data['r'] += 2
        self.n_relations = max(kg_data['r']) + 1
        self.n_users_entities = self.n_users + self.n_entities

        self.cf_train_data = (
            np.array(
                list(
                    map(
                        lambda d: d + self.n_entities,
                        self.cf_train_data[0]
                    )
                )
            ).astype(np.int32),
            self.cf_train_data[1].astype(np.int32)
        )

        self.cf_test_data = (
            np.array(
                list(
                    map(
                        lambda d: d + self.n_entities,
                        self.cf_test_data[0]
                    )
                )
            ).astype(np.int32),
            self.cf_test_data[1].astype(np.int32)
        )

        self.train_user_dict = {
            k + self.n_entities: np.unique(v).astype(np.int32)
            for k, v in self.train_user_dict.items()}
        self.test_user_dict = {
            k + self.n_entities: np.unique(v).astype(np.int32)
            for k, v in self.test_user_dict.items()}

        # add interactions to kg data
        cf2kg_train_data = pd.DataFrame(
            np.zeros((self.n_cf_train, 3), dtype=np.int32),
            columns=['h', 'r', 't'])
        cf2kg_train_data['h'] = self.cf_train_data[0]
        cf2kg_train_data['t'] = self.cf_train_data[1]

        inverse_cf2kg_train_data = pd.DataFrame(
            np.ones((self.n_cf_train, 3), dtype=np.int32),
            columns=['h', 'r', 't'])
        inverse_cf2kg_train_data['h'] = self.cf_train_data[1]
        inverse_cf2kg_train_data['t'] = self.cf_train_data[0]

        self.kg_train_data = pd.concat(
            [kg_data, cf2kg_train_data, inverse_cf2kg_train_data],
            ignore_index=True)
        self.n_kg_train = len(self.kg_train_data)

        # construct kg dict
        h_list = []
        t_list = []
        r_list = []

        self.train_kg_dict = collections.defaultdict(list)
        self.train_relation_dict = collections.defaultdict(list)

        for row in self.kg_train_data.iterrows():
            h, r, t = row[1]
            h_list.append(h)
            t_list.append(t)
            r_list.append(r)

            self.train_kg_dict[h].append((t, r))
            self.train_relation_dict[r].append((h, t))

        self.h_list = torch.LongTensor(h_list)
        self.t_list = torch.LongTensor(t_list)
        self.r_list = torch.LongTensor(r_list)

    def load_cache(self, path):
        assert os.path.exists(path)
        import pickle
        with open(path, mode="rb") as f:
            cache_dict = pickle.load(f)
        self.n_relations = cache_dict["n_relations"]
        self.n_entities = cache_dict["n_entities"]
        self.n_users_entities = cache_dict["n_users_entities"]

        self.cf_train_data = cache_dict["cf_train_data"]
        self.cf_test_data = cache_dict["cf_test_data"]

        self.train_user_dict = cache_dict["train_user_dict"]
        self.test_user_dict = cache_dict["test_user_dict"]

        self.kg_train_data = cache_dict["kg_train_data"]
        self.n_kg_train = cache_dict["n_kg_train"]

        self.train_kg_dict = cache_dict["train_kg_dict"]
        self.train_relation_dict = cache_dict["train_relation_dict"]

        self.h_list = cache_dict["h_list"]
        self.t_list = cache_dict["t_list"]
        self.r_list = cache_dict["r_list"]

        self.adjacency_dict = cache_dict["adjacency_dict"]
        self.laplacian_dict = cache_dict["laplacian_dict"]
        self.A_in = cache_dict["A_in"]

        self.type_of_scores = cache_dict["type_of_scores"]

    def store_cache(self, path):
        cache_dict = {
            "n_relations": self.n_relations,
            "n_entities": self.n_entities,
            "n_users_entities": self.n_users_entities,
            "cf_train_data": self.cf_train_data,
            "cf_test_data": self.cf_test_data,
            "train_user_dict": self.train_user_dict,
            "test_user_dict": self.test_user_dict,
            "kg_train_data": self.kg_train_data,
            "n_kg_train": self.n_kg_train,
            "train_kg_dict": self.train_kg_dict,
            "train_relation_dict": self.train_relation_dict,
            "h_list": self.h_list,
            "t_list": self.t_list,
            "r_list": self.r_list,
            "adjacency_dict": self.adjacency_dict,
            "laplacian_dict": self.laplacian_dict,
            "A_in": self.A_in,
            "type_of_scores": self.type_of_scores,
        }
        import pickle
        with open(path, mode="wb") as f:
            pickle.dump(cache_dict, f)

    def convert_coo2tensor(self, coo):
        values = coo.data
        indices = np.vstack((coo.row, coo.col))

        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = coo.shape
        return torch.sparse.FloatTensor(i, v, torch.Size(shape))

    def create_adjacency_dict(self):
        self.adjacency_dict = {}
        for r, ht_list in self.train_relation_dict.items():
            rows = [e[0] for e in ht_list]
            cols = [e[1] for e in ht_list]
            vals = [1] * len(rows)
            adj = sp.coo_matrix(
                (vals, (rows, cols)),
                shape=(self.n_users_entities, self.n_users_entities)
                )
            self.adjacency_dict[r] = adj

    def create_laplacian_dict(self):
        def symmetric_norm_lap(adj):
            rowsum = np.array(adj.sum(axis=1))

            d_inv_sqrt = np.power(rowsum, -0.5).flatten()
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0
            d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

            norm_adj = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
            return norm_adj.tocoo()

        def random_walk_norm_lap(adj):
            rowsum = np.array(adj.sum(axis=1))

            d_inv = np.power(rowsum, -1.0).flatten()
            d_inv[np.isinf(d_inv)] = 0
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            return norm_adj.tocoo()

        if self.laplacian_type == 'symmetric':
            norm_lap_func = symmetric_norm_lap
        elif self.laplacian_type == 'random-walk':
            norm_lap_func = random_walk_norm_lap
        else:
            raise NotImplementedError

        self.laplacian_dict = {}
        for r, adj in self.adjacency_dict.items():
            self.laplacian_dict[r] = norm_lap_func(adj)

        A_in = sum(self.laplacian_dict.values())
        self.A_in = self.convert_coo2tensor(A_in.tocoo())

    def print_info(self, logging):
        logging.info('n_users:           %d' % self.n_users)
        logging.info('n_items:           %d' % self.n_items)
        logging.info('n_entities:        %d' % self.n_entities)
        logging.info('n_users_entities:  %d' % self.n_users_entities)
        logging.info('n_relations:       %d' % self.n_relations)

        logging.info('n_h_list:          %d' % len(self.h_list))
        logging.info('n_t_list:          %d' % len(self.t_list))
        logging.info('n_r_list:          %d' % len(self.r_list))

        logging.info('n_cf_train:        %d' % self.n_cf_train)
        logging.info('n_cf_test:         %d' % self.n_cf_test)

        logging.info('n_kg_train:        %d' % self.n_kg_train)
