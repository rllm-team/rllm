import os
import os.path as osp
import pickle
import json
import csv
import warnings
from typing import Optional, Callable

import pandas as pd
import networkx as nx
import numpy as np
from numpy import ndarray
from scipy.sparse._csr import csr_matrix

import torch

from rllm.datasets.dataset import Dataset
from rllm.data.graph_data import GraphData
from rllm.utils.sparse import sparse_mx_to_torch_sparse_tensor
from rllm.utils.extract import extract_zip
from rllm.datasets.utils import index2mask, sanitize_name
from rllm.utils.download import download_url

warnings.filterwarnings("ignore", category=DeprecationWarning)


class TAPEDataset(Dataset):
    """The citation network datasets, include `cora` and `pubmed`,
    collected from paper `Harnessing Explanations: LLM-to-LM
    Interpreter for Enhanced Text-Attributed Graph Representation Learning
    <https://arxiv.org/abs/1603.08861>`__ paper.

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
        use_text (bool):
            If set to `False`, original text will not be loaded.
        use_gpt (bool):
            If set to `False`, gpt explanations will not be loaded.
        use_preds (bool):
            If set to `False`, pesudo-labels annotated by gpt
            will not be loaded.
        topk (int):
            the top-k pesudo-labels to be loaded, the default value is 5.
        forced_reload (bool):
            If set to `True`, this dataset will be re-process again.
    """

    urls = {
        "original text": {
            "cora": "https://drive.usercontent.google.com/download?id=1hxE0OPR7VLEHesr48WisynuoNMhXJbpl&confirm=t",  # noqa
            "pubmed": "https://drive.usercontent.google.com/download?id=1sYZX-jP6H8OkopVa9cp8-KXdEti5ki_W&confirm=t",  # noqa
        },
        "LLM responses": {
            "cora": "https://drive.usercontent.google.com/download?id=1tSepgcztiNNth4kkSR-jyGkNnN7QDYax&confirm=t",  # noqa
            "pubmed": "https://drive.usercontent.google.com/download?id=166waPAjUwu7EWEvMJ0heflfp0-4EvrZS&confirm=t",  # noqa
        },
        "GPT predictions": {
            "cora": "https://raw.githubusercontent.com/XiaoxinHe/TAPE/main/gpt_preds/cora.csv",  # noqa
            "pubmed": "https://raw.githubusercontent.com/XiaoxinHe/TAPE/main/gpt_preds/pubmed.csv",  # noqa
        },
    }

    def __init__(
        self,
        cached_dir: str,
        file_name: str,
        transform: Optional[Callable] = None,
        use_text: Optional[bool] = True,
        use_gpt: Optional[bool] = True,
        use_preds: Optional[bool] = True,
        topk: Optional[int] = 5,
        force_reload: Optional[bool] = False,
    ):
        self.name = file_name.lower()
        assert self.name in ["cora", "pubmed"]
        root = os.path.join(cached_dir, f"TAPE_{self.name}")

        self.use_text = use_text
        self.use_gpt = use_gpt
        self.use_preds = use_preds
        self.topk = topk

        super().__init__(root, force_reload=force_reload)
        self.data_list = [GraphData.load(self.processed_paths[0])]
        self._filter_elements()

        self.transform = transform
        if self.transform is not None:
            self.data_list[0] = self.transform(self.data_list[0])

    def _filter_elements(self):
        data = self.data_list[0]
        if not self.use_text:
            del data.text
        if not self.use_gpt:
            del data.gpt
        if not self.use_preds:
            del data.gpt_preds

        # split data
        num_nodes = data.num_nodes
        node_id = np.arange(num_nodes)
        np.random.shuffle(node_id)

        train_id = np.sort(node_id[: int(num_nodes * 0.6)])
        val_id = np.sort(node_id[int(num_nodes * 0.6) : int(num_nodes * 0.8)])
        test_id = np.sort(node_id[int(num_nodes * 0.8) :])

        data.train_mask = index2mask(train_id, num_nodes)
        data.val_mask = index2mask(val_id, num_nodes)
        data.test_mask = index2mask(test_id, num_nodes)

    @property
    def raw_filenames(self):

        if self.name == "cora":
            filenames = ["Cora", "cora_orig"]
        elif self.name == "pubmed":
            filenames = ["PubMed", "PubMed_orig"]
        else:
            filenames = [self.name, f"{self.name}_orig"]
        return filenames + [f"{self.name}.csv"]

    @property
    def gpt_folder(self):
        if self.name == "cora":
            return osp.join(self.raw_dir, "Cora")
        if self.name == "pubmed":
            return osp.join(self.raw_dir, "PubMed")

        return osp.join(self.raw_dir, self.name)

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

        if self.name == "cora":
            data = self._get_raw_text_cora()
        elif self.name == "pubmed":
            data = self._get_raw_text_pubmed()

        gpt = []
        num_nodes = data.num_nodes
        gpt_folder = self.gpt_folder
        for i in range(num_nodes):
            filepath = osp.join(gpt_folder, f"{i}.json")
            with open(filepath, "r") as file:
                json_data = json.load(file)
                content = json_data["choices"][0]["message"]["content"]
                gpt.append(content)
        data.gpt = gpt

        preds = []
        preds_path = osp.join(self.raw_dir, f"{self.name}.csv")
        with open(preds_path, "r") as file:
            reader = csv.reader(file)
            for row in reader:
                inner_list = []
                for value in row:
                    inner_list.append(int(value))
                preds.append(inner_list)

        pl = torch.zeros(len(preds), self.topk, dtype=torch.long)
        for i, pred in enumerate(preds):
            pl[i][: len(pred)] = torch.tensor(pred[: self.topk], dtype=torch.long) + 1
        data.gpt_preds = pl

        data.save(self.processed_paths[0])

    def download(self):
        r"""
        download data from url to './cached_dir/{dataset}/raw/'.
        """
        os.makedirs(self.raw_dir, exist_ok=True)

        path_text = download_url(
            self.urls["original text"][self.name], self.raw_dir, f"{self.name}_orig.zip"
        )
        path_llm = download_url(
            self.urls["LLM responses"][self.name], self.raw_dir, f"{self.name}.zip"
        )
        path_preds = download_url(  # noqa
            self.urls["GPT predictions"][self.name], self.raw_dir, f"{self.name}.csv"
        )

        extract_zip(path_text, self.raw_dir)
        extract_zip(path_llm, self.raw_dir)
        os.remove(path_text)
        os.remove(path_llm)

    def item(self):
        return self.data_list[0]

    def __len__(self):
        return 1

    def __getitem__(self, index: int):
        if index != 0:
            raise IndexError
        return self.data_list[index]

    ##########
    # cora #
    ##########
    def _get_raw_text_cora(self):
        path = osp.join(self.raw_dir, "cora_orig", "cora")
        idx_features_labels = np.genfromtxt(
            "{}.content".format(path), dtype=np.dtype(str)
        )
        data_X = idx_features_labels[:, 1:-1].astype(np.float32)
        labels = idx_features_labels[:, -1]
        class_map = {
            x: i
            for i, x in enumerate(
                [
                    "Case_Based",
                    "Genetic_Algorithms",
                    "Neural_Networks",
                    "Probabilistic_Methods",
                    "Reinforcement_Learning",
                    "Rule_Learning",
                    "Theory",
                ]
            )
        }
        data_Y = np.array([class_map[i] for i in labels])
        data_citeid = idx_features_labels[:, 0]
        idx = np.array(data_citeid, dtype=np.dtype(str))
        idx_map = {j: i for i, j in enumerate(idx)}
        edges_unordered = np.genfromtxt("{}.cites".format(path), dtype=np.dtype(str))
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten()))).reshape(
            edges_unordered.shape
        )
        data_edges = np.array(edges[~(edges == None).max(1)], dtype="int")
        data_edges = np.vstack((data_edges, np.fliplr(data_edges)))
        data_edges = np.unique(data_edges, axis=0)

        x = torch.tensor(data_X).float()
        y = torch.tensor(data_Y).long()
        G = nx.from_edgelist(data_edges)
        adj_sp = sparse_mx_to_torch_sparse_tensor(nx.to_scipy_sparse_array(G))

        # parse text
        base_dir = osp.join(self.raw_dir, "cora_orig", "mccallum", "cora")
        with open(osp.join(base_dir, "papers")) as f:
            lines = f.readlines()
        pid_filename = {}
        for line in lines:
            pid = line.split("\t")[0]
            fn = line.split("\t")[1]
            if osp.sep == "\\":
                fn = sanitize_name(fn, osp.sep)
            pid_filename[pid] = fn

        path = osp.join(base_dir, "extractions")
        text = []
        for pid in data_citeid:
            fn = pid_filename[pid]
            with open(osp.join(path, fn), "r") as f:
                lines = f.read().splitlines()

            for line in lines:
                if "Title:" in line:
                    ti = line
                if "Abstract:" in line:
                    ab = line
            text.append(ti + "\n" + ab)

        data = GraphData(x=x, y=y, adj=adj_sp, text=text)
        return data

    def _get_raw_text_pubmed(self):
        path = osp.join(self.raw_dir, "PubMed_orig", "data")
        n_nodes = 19717
        n_features = 500

        data_X = np.zeros((n_nodes, n_features), dtype="float32")
        data_Y = [None] * n_nodes
        data_pubid = [None] * n_nodes
        data_edges = []

        paper_to_index = {}
        feature_to_index = {}

        # parse nodes
        with open(osp.join(path, "Pubmed-Diabetes.NODE.paper.tab"), "r") as node_file:
            # first two lines are headers
            node_file.readline()
            node_file.readline()

            k = 0

            for i, line in enumerate(node_file.readlines()):
                items = line.strip().split("\t")

                paper_id = items[0]
                data_pubid[i] = paper_id
                paper_to_index[paper_id] = i

                # label=[1,2,3]
                label = int(items[1].split("=")[-1]) - 1  # subtract 1 to zero-count
                data_Y[i] = label

                # f1=val1 \t f2=val2 \t ... \t fn=valn summary=...
                features = items[2:-1]
                for feature in features:
                    parts = feature.split("=")
                    fname = parts[0]
                    fvalue = float(parts[1])

                    if fname not in feature_to_index:
                        feature_to_index[fname] = k
                        k += 1

                    data_X[i, feature_to_index[fname]] = fvalue

        # parse graph
        with open(
            osp.join(path, "Pubmed-Diabetes.DIRECTED.cites.tab"), "r"
        ) as edge_file:
            # first two lines are headers
            edge_file.readline()
            edge_file.readline()

            for i, line in enumerate(edge_file.readlines()):

                # edge_id \t paper:tail \t | \t paper:head
                items = line.strip().split("\t")

                edge_id = items[0]  # noqa

                tail = items[1].split(":")[-1]
                head = items[3].split(":")[-1]

                if head != tail:
                    data_edges.append((paper_to_index[head], paper_to_index[tail]))
                    data_edges.append((paper_to_index[tail], paper_to_index[head]))

        from sklearn.preprocessing import normalize

        data_X = normalize(data_X, norm="l1")
        x = torch.tensor(data_X).float()
        y = torch.tensor(data_Y).long()
        G = nx.from_edgelist(data_edges)
        adj_sp = sparse_mx_to_torch_sparse_tensor(nx.to_scipy_sparse_array(G))

        with open(osp.join(self.raw_dir, "PubMed_orig", "pubmed.json")) as f:
            pubmed = json.load(f)
            df_pubmed = pd.DataFrame.from_dict(pubmed)

            AB = df_pubmed["AB"].fillna("")
            TI = df_pubmed["TI"].fillna("")
            text = []
            for ti, ab in zip(TI, AB):
                t = "Title: " + ti + "\n" + "Abstract: " + ab
                text.append(t)

        data = GraphData(x=x, y=y, adj=adj_sp, text=text)
        return data
