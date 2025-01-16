import os
import os.path as osp
import pickle
import types
import warnings
from typing import Optional, Callable

import torch

from rllm.datasets.dataset import Dataset
from rllm.data.graph_data import GraphData
from rllm.utils.download import download_url
from rllm.data.storage import BaseStorage

warnings.filterwarnings("ignore", category=DeprecationWarning)


class TAGDataset(Dataset):
    """Three text-attributed-graph datasets, including
    `cora` from `Automating the Construction of Internet Portals
    <https://link.springer.com/content/pdf/10.1023/A:1009953814988.pdf>`__,
    `pubmed` from `Collective Classification in Network Data
    <https://ojs.aaai.org/aimagazine/index.php/aimagazine/article/view/2157>`__
    and `citeseer` from `CiteSeer: an automatic citation
    indexing system <https://dl.acm.org/doi/10.1145/276675.276685>`__ paper.
    This dataset also contains cached LLM predictions and confidences
    provided by the paper `Label-free Node Classification on Graphs
    with Large Language Models (LLMS) <https://arxiv.org/abs/2310.04668>`__.

    Args:
        cached_dir (str):
            Root directory where dataset should be saved.
        file_name (str):
            The name of dataset, *e.g.*, `cora` and `pubmed`.
        transform (callable, optional):
            A function/transform that takes in an
            `GraphData` object and returns a transformed version.
            The data object will be transformed before every access.
            (default: `None`)
        use_preds (bool):
            If set to `False`, cached pesudo-labels annotated
            by gpt will not be loaded.
        forced_reload (bool):
            If set to `True`, this dataset will be re-process again.
    """

    urls = {
        "text": {
            "cora": "https://drive.usercontent.google.com/download?id=10iBkU36HGc9mVPOWUofVyC_9cJ-U1-xl&confirm=t",  # noqa
            "pubmed": "https://drive.usercontent.google.com/download?id=1hcqnmKXv4dk060k6VjmW66e8etPYcOW9&confirm=t",  # noqa
            "citeseer": "https://drive.usercontent.google.com/download?id=1JRJHDiKFKiUpozGqkWhDY28E6F4v5n7l&confirm=t",  # noqa
        },
        "pred": {
            "cora": "https://drive.usercontent.google.com/download?id=1jAZH9daUjg0ce9O4IqitPA0coWCJKcMs&confirm=t",  # noqa
            "pubmed": "https://drive.usercontent.google.com/download?id=1d7saxT6Uc5sA4UpZ1ujky_Ns9vChgw9L&confirm=t",  # noqa
            "citeseer": "https://drive.usercontent.google.com/download?id=1k4L-NPxbrd9hiTehspjd0Ue6NOT-PRWi&confirm=t",  # noqa
        },
    }

    def __init__(
        self,
        cached_dir: str,
        file_name: str,
        transform: Optional[Callable] = None,
        use_cache: bool = True,
        force_reload: Optional[bool] = False,
    ):
        self.name = file_name.lower()
        assert self.name in ["cora", "pubmed", "citeseer"]
        root = os.path.join(cached_dir, f"LLMGNN_{self.name}")

        self.use_cache = use_cache

        super().__init__(root, force_reload=force_reload)
        self.data_list = [GraphData.load(self.processed_paths[0])]

        self.transform = transform
        if self.transform is not None:
            self.data_list[0] = self.transform(self.data_list[0])

    @property
    def raw_filenames(self):
        filenames = [f"{self.name}_fixed_sbert.pt", f"{self.name}^cache^consistency.pt"]
        return filenames

    @property
    def processed_filenames(self):
        return ["data.pt"]

    def process(self):
        r"""
        process data and save to './cached_dir/{dataset}/processed/'.
        """
        os.makedirs(self.processed_dir, exist_ok=True)

        data = self._get_raw_text()

        if self.use_cache:
            filepath = osp.join(self.raw_dir, f"{self.name}^cache^consistency.pt")
            cache_data = torch.load(filepath, weights_only=False)
            data.pl = cache_data["pred"]
            data.conf = cache_data["conf"]
            data.cache_mask = data.pl >= 0
        data.save(self.processed_paths[0])

    def download(self):
        r"""
        download data from url to './cached_dir/{dataset}/raw/'.
        """
        os.makedirs(self.raw_dir, exist_ok=True)

        path_text = download_url(  # noqa
            self.urls["text"][self.name], self.raw_dir, f"{self.name}_fixed_sbert.pt"
        )
        path_pred = download_url(  # noqa
            self.urls["pred"][self.name],
            self.raw_dir,
            f"{self.name}^cache^consistency.pt",
        )

    def item(self):
        return self.data_list[0]

    def __len__(self):
        return 1

    def __getitem__(self, index: int):
        if index != 0:
            raise IndexError
        return self.data_list[index]

    def _get_raw_text(self):

        class CustomUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if "torch_geometric" in module:
                    if name == "GlobalStorage":
                        return BaseStorage
                    else:
                        return types.SimpleNamespace
                return super().find_class(module, name)

        custom_pickle_module = types.ModuleType("custom_pickle_module")
        custom_pickle_module.Unpickler = CustomUnpickler
        custom_pickle_module.load = pickle.load

        path = osp.join(self.raw_dir, f"{self.name}_fixed_sbert.pt")
        raw_data = torch.load(
            path, pickle_module=custom_pickle_module, weights_only=False
        )._store
        num_nodes = raw_data.edge_index.max().item() + 1
        adj = torch.sparse_coo_tensor(
            indices=raw_data.edge_index,
            values=torch.ones(raw_data.edge_index.size(1)),
            size=(num_nodes, num_nodes),
        )
        data = GraphData(x=raw_data.x, y=raw_data.y, adj=adj, text=raw_data.raw_texts)
        data.label_names = raw_data.label_names
        return data
