import os
import os.path as osp
import pickle
from typing import Optional, Callable

import numpy as np
from numpy import ndarray
from scipy.sparse._csr import csr_matrix
import networkx as nx
import warnings

import torch

# import sys
# sys.path.append('../')
from rllm.datasets.dataset import Dataset
from rllm.data.graph_data import GraphData
from rllm.utils.sparse import sparse_mx_to_torch_sparse_tensor
from rllm.datasets.utils import index2mask
from rllm.utils.download import download_url

warnings.filterwarnings("ignore", category=DeprecationWarning)


class PlanetoidDataset(Dataset):
    r"""The citation network datasets from the `Revisiting Semi-Supervised
    Learning with Graph Embeddings <https://arxiv.org/abs/1603.08861>`__ paper,
    which include :obj:`"Cora"`, :obj:`"CiteSeer"` and :obj:`"PubMed"`.
    Nodes represent documents and edges represent citation links.

    Args:
        cached_dir (str): Root directory where dataset should be saved.
        file_name (str): The name of dataset, *e.g.*, `cora`, `citeseer`
            and `pubmed`.
        transform (callable, optional): A function/transform that takes in an
            `GraphData` object and returns a transformed version.
            The data object will be transformed before every access.
            (default: `None`)
        split (str, optional): The type of dataset split (`public`,
            `full`, `geom-gcn`, `random`).
            If set to `public`, the split will be the public fixed split
            from the `Revisiting Semi-Supervised Learning with Graph
            Embeddings <https://arxiv.org/abs/1603.08861>`__ paper.
            If set to `full`, all nodes except those in the validation
            and test sets will be used for training (as in the
            `FastGCN: Fast Learning with Graph Convolutional Networks via
            Importance Sampling <https://arxiv.org/abs/1801.10247>`__ paper).
            If set to `geom-gcn`, the 10 public fixed splits from the
            `Geom-GCN: Geometric Graph Convolutional Networks
            <https://openreview.net/forum?id=S1e2agrFvS>`__ paper are given.
            If set to `random`, train, validation, and test sets will be
            randomly generated, according to `num_train_per_class`,
            `num_val` and `num_test`. (default: `public`)
        num_train_per_class (int, optional): The number of training samples
            per class in case of `random` split. (default: 20)
        num_val (int, optional): The number of validation samples in case of
            `random` split. (default: 500)
        num_test (int, optional): The number of test samples in case of
            `random` split. (default: 1000)
        forced_reload (bool): If set to `True`, this dataset will be
            re-process again.

    .. parsed-literal::

        Statics:
        Name        Cora    CiteSeer    PubMed
        nodes       2708    3327        19717
        edges       10556   9104        88648
        features    1433    3703        500
        classes     7       6           3
    """

    url = "https://github.com/kimiyoung/planetoid/raw/master/data"
    geom_gcn_url = (
        "https://raw.githubusercontent.com/graphdml-uiuc-jlu/geom-gcn/master"  # noqa
    )

    def __init__(
        self,
        cached_dir: str,
        file_name: str,
        transform: Optional[Callable] = None,
        split: str = "public",
        num_train_per_class: int = 20,
        num_val: int = 500,
        num_test: int = 1000,
        force_reload: Optional[bool] = False,
    ):

        self.name = file_name.lower()
        assert self.name in ["cora", "citeseer", "pubmed"]
        self.split = split.lower()
        assert self.split in ["public", "full", "geom-gcn", "random"]

        root = osp.join(cached_dir, self.name)
        if self.split == "geom-gcn":
            root = osp.join(root, "geom-gcn")
        super().__init__(root, force_reload=force_reload)

        self.data_list = [GraphData.load(self.processed_paths[0])]

        if self.split == "full":
            data = self.data_list[0]
            data.train_mask.fill_(True)
            data.train_mask[data.val_mask | data.test_mask] = False
        elif split == "random":
            data = self.data_list[0]
            data.train_mask.fill_(False)
            for c in range(data.num_classes):
                idx = (data.y == c).nonzero(as_tuple=False).view(-1)
                idx = idx[torch.randperm(idx.size(0))[:num_train_per_class]]
                data.train_mask[idx] = True

            remaining = (~data.train_mask).nonzero(as_tuple=False).view(-1)
            remaining = remaining[torch.randperm(remaining.size(0))]

            data.val_mask.fill_(False)
            data.val_mask[remaining[:num_val]] = True

            data.test_mask.fill_(False)
            data.test_mask[remaining[num_val : num_val + num_test]] = True

        self.transform = transform
        if self.transform is not None:
            self.data_list[0] = self.transform(self.data_list[0])

    @property
    def raw_filenames(self):
        suffix = ["x", "tx", "allx", "y", "ty", "ally", "graph", "test.index"]
        return [f"ind.{self.name}.{s}" for s in suffix]

    @property
    def processed_filenames(self):
        return ["data.pt"]

    def _load_raw_file(self, filename: str):
        r"""
        load data from './cached_dir/{dataset}/raw/'
        """
        filepath = osp.join(self.raw_dir, filename)
        if "test.index" in filename:
            with open(filepath, "r") as f:
                lines = f.readlines()
                out = [int(line.strip("\n")) for line in lines]
                out = torch.as_tensor(out, dtype=torch.long)
        else:
            with open(filepath, "rb") as f:
                content = pickle.load(f, encoding="latin1")
                if isinstance(content, csr_matrix):
                    out = content.todense()
                    out = torch.from_numpy(out).float()
                elif isinstance(content, ndarray):
                    out = torch.from_numpy(content).float()
                else:
                    out = content
        return out

    def process(self):
        r"""
        process data and save to './cached_dir/{dataset}/processed/'.
        """
        os.makedirs(self.processed_dir, exist_ok=True)

        items = [self._load_raw_file(filename) for filename in self.raw_filenames]
        x, tx, allx, y, ty, ally, graph, test_index = items
        train_index = torch.arange(x.shape[0], dtype=torch.long)
        val_index = torch.arange(x.shape[0], x.shape[0] + 500, dtype=torch.long)
        sorted_test_index, _ = test_index.sort()

        if self.name == "citeseer":
            # For citeseer, there are some isolated nodes.
            # We should find them and add them as
            # zero-vector in the right position.
            min_index, max_index = sorted_test_index[0], sorted_test_index[-1]

            tx_ext = torch.zeros(max_index - min_index + 1, tx.shape[1], dtype=tx.dtype)
            tx_ext[sorted_test_index - min_index, :] = tx
            ty_ext = torch.zeros(max_index - min_index + 1, ty.shape[1], dtype=ty.dtype)
            ty_ext[sorted_test_index - min_index, :] = ty

            tx, ty = tx_ext, ty_ext

        x = torch.cat([allx, tx], dim=0)
        x[test_index] = x[sorted_test_index]
        y = torch.cat([ally, ty], dim=0).max(dim=1)[1]
        y[test_index] = y[sorted_test_index]

        if self.split == "geom-gcn":
            train_masks, val_masks, test_masks = [], [], []
            for i in range(10):
                name = f"{self.name.lower()}_split_0.6_0.2_{i}.npz"
                splits = np.load(osp.join(self.raw_dir, name))
                train_masks.append(torch.from_numpy(splits["train_mask"]))
                val_masks.append(torch.from_numpy(splits["val_mask"]))
                test_masks.append(torch.from_numpy(splits["test_mask"]))
            train_mask = torch.stack(train_masks, dim=1)
            val_mask = torch.stack(val_masks, dim=1)
            test_mask = torch.stack(test_masks, dim=1)
        else:
            train_mask = index2mask(train_index, x.shape[0])
            val_mask = index2mask(val_index, x.shape[0])
            test_mask = index2mask(test_index, x.shape[0])

        G = nx.from_dict_of_lists(graph)
        adj_sp = sparse_mx_to_torch_sparse_tensor(nx.to_scipy_sparse_array(G))
        data = GraphData(
            x, y, adj_sp, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask
        )
        data.save(self.processed_paths[0])

    def download(self):
        r"""
        download data from url to './cached_dir/{dataset}/raw/'.
        """
        os.makedirs(self.raw_dir, exist_ok=True)
        for filename in self.raw_filenames:
            target_url = f"{self.url}/{filename}"
            download_url(target_url, self.raw_dir, filename)

        if self.split == "geom-gcn":
            for i in range(10):
                url = f"{self.geom_gcn_url}/splits/{self.name.lower()}"
                download_url(f"{url}_split_0.6_0.2_{i}.npz", self.raw_dir)

    def item(self):
        return self.data_list[0]

    def __len__(self):
        return 1

    def __getitem__(self, index: int):
        if index != 0:
            raise IndexError
        return self.data_list[index]
