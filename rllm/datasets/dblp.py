import os
import os.path as osp
from itertools import product
from typing import Optional, Callable

import numpy as np
import scipy.sparse as sp
import torch

from rllm.datasets.dataset import Dataset
from rllm.data.graph_data import HeteroGraphData
from rllm.utils.graph_utils import sparse_mx_to_torch_sparse_tensor
from rllm.utils.download import download_url
from rllm.utils.extract import extract_zip
from rllm.datasets.utils import index2mask


class DBLP(Dataset):
    r"""DBLP is a heterogeneous graph containing four types of entities,
    as collected in the `MAGNN: Metapath Aggregated Graph Neural Network for
    Heterogeneous Graph Embedding <https://arxiv.org/abs/2002.01680>`__ paper.

    The authors are divided into four research areas (database, data mining,
    artificial intelligence, information retrieval).
    Each author is described by a bag-of-words representation of their paper
    keywords.

    Args:
        cached_dir (str): Root directory where dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            `HeteroGraphData` object and returns a transformed version.
            The data object will be transformed before every access.
        forced_reload (bool): If set to `True`, this dataset
            will be re-process again.

    .. parsed-literal::

        Statics:
        Name    authors     papers      terms       conferences
        nodes   4,057       14,328      7,723       20

    """
    url = 'https://www.dropbox.com/s/yh4grpeks87ugr2/DBLP_processed.zip?dl=1'

    def __init__(self,
                 cached_dir: str,
                 transform: Optional[Callable] = None,
                 forced_reload: Optional[bool] = False):
        self.name = 'dblp'
        root = os.path.join(cached_dir, self.name)
        super().__init__(root, force_reload=forced_reload)
        self.data_list = [HeteroGraphData.load(self.processed_paths[0])]

        self.transform = transform
        if self.transform is not None:
            self.data_list[0] = self.transform(self.data_list[0])

    @property
    def raw_filenames(self):
        return [
            'adjM.npz', 'features_0.npz', 'features_1.npz', 'features_2.npy',
            'labels.npy', 'node_types.npy', 'train_val_test_idx.npz'
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
        node_types = ['author', 'paper', 'term', 'conference']
        for i, node_type in enumerate(node_types[:2]):
            x = sp.load_npz(osp.join(self.raw_dir, f'features_{i}.npz'))
            data[node_type].x = torch.from_numpy(x.todense()).to(torch.float)

        x = np.load(osp.join(self.raw_dir, 'features_2.npy'))
        data['term'].x = torch.from_numpy(x).to(torch.float)

        node_type_idx = np.load(osp.join(self.raw_dir, 'node_types.npy'))
        node_type_idx = torch.from_numpy(node_type_idx).to(torch.long)
        data['conference'].num_nodes = int((node_type_idx == 3).sum())

        y = np.load(osp.join(self.raw_dir, 'labels.npy'))
        data['author'].y = data.y = torch.from_numpy(y).to(torch.long)

        split = np.load(osp.join(self.raw_dir, 'train_val_test_idx.npz'))
        data.train_mask = index2mask(
            torch.from_numpy(split['train_idx']).to(torch.long),
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
        N_a = data['author'].num_nodes
        N_p = data['paper'].num_nodes
        N_t = data['term'].num_nodes
        N_c = data['conference'].num_nodes
        s['author'] = (0, N_a)
        s['paper'] = (N_a, N_a + N_p)
        s['term'] = (N_a + N_p, N_a + N_p + N_t)
        s['conference'] = (N_a + N_p + N_t, N_a + N_p + N_t + N_c)

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
        path = download_url(self.url, self.raw_dir, 'DBLP_processed.zip')
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
