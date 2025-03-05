from typing import Optional, List
import os
import os.path as osp

import numpy as np
import pandas as pd
import torch

from rllm.types import ColType
from rllm.data.table_data import TableData
from rllm.datasets.dataset import Dataset
from rllm.utils.download import download_url
from rllm.utils.extract import extract_zip


class TACM12KDataset(Dataset):
    r"""TACM12KDataset is a multi-table relational dataset containing 4 tables,
    as collected in the `rLLM: Relational Table Learning with LLMs
    <https://arxiv.org/abs/2407.20157>`__ paper.

    It includes four tables: papers, authors, citations and writings.
    The papers table includes publication information of papers.
    The authors table includes author information.
    The citations table includes citation (i.e., <paper, paper>) information
    between papers.
    The writings table includes <author, write, paper> relationship
    between authors and papers.
    The default task is to predict the conference of papers.

    Args:
        cached_dir (str): Root directory where dataset should be saved.
        forced_reload (bool): If set to `True`, this dataset will be
            re-process again.

    .. parsed-literal::

        Table1: papers
        ---------------
            Statics:
            Name        Papers      Features
            Size        12,499      5

        Table2: authors
        ------------------
            Statics:
            Name        Authors     Features
            Size        17,431      3

        Table3: citations
        ------------------
            Statics:
            Name        Citations   Features
            edges       30,789      2

        Table4: writings
        ------------------
            Statics:
            Name        Writings    Features
            edges       37,055      2
    """

    url = "https://raw.githubusercontent.com/rllm-project/rllm_datasets/refs/heads/main/sjtutables/tacm12k.zip"  # noqa

    def __init__(self, cached_dir: str, force_reload: Optional[bool] = False) -> None:
        self.name = "tacm12k"
        root = os.path.join(cached_dir, self.name)
        super().__init__(root, force_reload=force_reload)

        # Table_ACM12K data_list:
        # 0: papers_table
        # 1: authors_table
        # 2: citations_table
        # 3: writings_table
        # 4: paper_embeddings
        # 5: author_embeddings
        self.data_list: List[TableData] = [
            TableData.load(self.processed_paths[0]),
            TableData.load(self.processed_paths[1]),
            TableData.load(self.processed_paths[2]),
            TableData.load(self.processed_paths[3]),
            torch.from_numpy(np.load(osp.join(self.raw_dir, "paper_embeddings.npy"))),
            torch.from_numpy(np.load(osp.join(self.raw_dir, "author_embeddings.npy"))),
        ]

    @property
    def raw_filenames(self):
        return [
            "papers.csv",
            "authors.csv",
            "citations.csv",
            "writings.csv",
            "masks.pt",
        ]

    @property
    def processed_filenames(self):
        return [
            "paper_data.pt",
            "authors_data.pt",
            "citations_data.pt",
            "writings_data.pt",
        ]

    def process(self):
        r"""
        process data and save to './cached_dir/{dataset}/processed/'.
        """
        os.makedirs(self.processed_dir, exist_ok=True)
        # papers Data
        path = osp.join(self.raw_dir, self.raw_filenames[0])
        paper_df = pd.read_csv(path, index_col=["paper_id"])
        col_types = {
            "year": ColType.CATEGORICAL,
            "conference": ColType.CATEGORICAL,
            "title": ColType.CATEGORICAL,
            "abstract": ColType.CATEGORICAL,
        }

        # Create masks
        masks_path = osp.join(self.raw_dir, self.raw_filenames[4])
        masks = torch.load(masks_path, weights_only=False)
        TableData(
            df=paper_df,
            col_types=col_types,
            target_col="conference",
            train_mask=masks["train_mask"],
            val_mask=masks["val_mask"],
            test_mask=masks["test_mask"],
        ).save(self.processed_paths[0])

        # authors Data
        path = osp.join(self.raw_dir, self.raw_filenames[1])
        author_df = pd.read_csv(path, index_col=["author_id"])
        col_types = {
            "name": ColType.CATEGORICAL,
            "firm": ColType.CATEGORICAL,
        }
        TableData(df=author_df, col_types=col_types).save(self.processed_paths[1])

        # cite Data
        path = osp.join(self.raw_dir, self.raw_filenames[2])
        cite_df = pd.read_csv(path)
        col_types = {
            "paper_id": ColType.NUMERICAL,
            "paper_id_cited": ColType.NUMERICAL,
        }
        TableData(df=cite_df, col_types=col_types).save(self.processed_paths[2])

        # cite Data
        path = osp.join(self.raw_dir, self.raw_filenames[3])
        pa_df = pd.read_csv(path)
        col_types = {
            "paper_id": ColType.NUMERICAL,
            "author_id": ColType.NUMERICAL,
        }
        TableData(df=pa_df, col_types=col_types).save(self.processed_paths[3])

    def download(self):
        os.makedirs(self.raw_dir, exist_ok=True)
        path = download_url(self.url, self.raw_dir, self.name + ".zip")
        extract_zip(path, self.raw_dir)
        os.remove(path)

    def __len__(self):
        return 4

    def __getitem__(self, index: int):
        if index < 0 or index > self.__len__():
            raise IndexError
        return self.data_list[index]
