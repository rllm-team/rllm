import os
import os.path as osp
from itertools import product
from typing import Optional, Callable

import numpy as np
import scipy.sparse as sp
import torch

# import sys
# sys.path.append('../')
from rllm.datasets.dataset import Dataset
from rllm.data.graph_data import HeteroGraphData
from rllm.utils.sparse import sparse_mx_to_torch_sparse_tensor
from rllm.utils.extract import extract_zip
from rllm.datasets.utils import index2mask
from rllm.utils.download import download_url


class IMDB(Dataset):
    r"""IMDB is a heterogeneous graph containing three types of entities,
    as collected in the `MAGNN: Metapath Aggregated Graph Neural Network for
    Heterogeneous Graph Embedding <https://arxiv.org/abs/2002.01680>`__ paper.

    The movies are divided into three classes (action, comedy, drama) according
    to their genre.
    Movie features correspond to elements of a bag-of-words representation of
    its plot keywords.

    Args:
        cached_dir(str): Root directory where dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            `HeteroGraphData` object and returns a transformed version.
            The data object will be transformed before every access.
            (default: `None`)
        forced_reload (bool): If set to `True`, this dataset will be
            re-process again.

    .. parsed-literal::

        Statics:
        Name    movie     actors      directors
        nodes   4,278       5,257      2,081
    """
    url = 'https://www.dropbox.com/s/g0btk9ctr1es39x/IMDB_processed.zip?dl=1'

    def __init__(self,
                 cached_dir: str,
                 transform: Optional[Callable] = None,
                 force_reload: Optional[bool] = False):
        self.name = 'imdb'
        root = os.path.join(cached_dir, self.name)
        super().__init__(root, force_reload=force_reload)
        self.data_list = [HeteroGraphData.load(self.processed_paths[0])]

        self.transform = transform
        if self.transform is not None:
            self.data_list[0] = self.transform(self.data_list[0])

    @property
    def raw_filenames(self):
        return [
            'adjM.npz', 'features_0.npz', 'features_1.npz', 'features_2.npz',
            'labels.npy', 'train_val_test_idx.npz'
        ]

    @property
    def processed_filenames(self):
        return ['data.pt']

    def process(self):
        r"""
        process data and save to './cached_dir/{dataset}/processed/'.
        """

        os.makedirs(self.processed_dir, exist_ok=True)

        data = HeteroGraphData()
        node_types = ['movie', 'director', 'actor']
        for i, node_type in enumerate(node_types):
            x = sp.load_npz(osp.join(self.raw_dir, f'features_{i}.npz'))
            data[node_type].x = torch.from_numpy(x.todense()).to(torch.float)

        y = np.load(osp.join(self.raw_dir, 'labels.npy'))
        data['movie'].y = data.y = torch.from_numpy(y).to(torch.long)
        split = np.load(osp.join(self.raw_dir, 'train_val_test_idx.npz'))
        data.train_mask = index2mask(torch.from_numpy(
            split['train_idx']).to(torch.long),
            len(y)
        )
        data.val_mask = index2mask(
            torch.from_numpy(split['val_idx']).to(torch.long),
            len(y)
        )
        data.test_mask = index2mask(
            torch.from_numpy(split['test_idx']).to(torch.long),
            len(y)
        )

        s = {}
        N_m = data['movie'].num_nodes
        N_d = data['director'].num_nodes
        N_a = data['actor'].num_nodes
        s['movie'] = (0, N_m)
        s['director'] = (N_m, N_m + N_d)
        s['actor'] = (N_m + N_d, N_m + N_d + N_a)

        A = sp.load_npz(osp.join(self.raw_dir, 'adjM.npz'))
        for src, dst in product(node_types, node_types):
            A_sub = A[s[src][0]:s[src][1], s[dst][0]:s[dst][1]].tocoo()
            if A_sub.nnz > 0:
                data[src, dst].adj = sparse_mx_to_torch_sparse_tensor(A_sub)
        # data.adj = sparse_mx_to_torch_sparse_tensor(A)
        data.save(self.processed_paths[0])

    def download(self):
        r"""
        download data from url to './cached_dir/{dataset}/raw/'.
        """
        os.makedirs(self.raw_dir, exist_ok=True)
        path = download_url(self.url, self.raw_dir, 'IMDB_processed.zip')
        extract_zip(path, self.raw_dir)
        os.remove(path)

    def item(self):
        return self.data_list[0]

    def __len__(self):
        return 1

    def __getitem__(self, index: int):
        if index != 0:
            raise IndexError
        return self.data_list[index]
