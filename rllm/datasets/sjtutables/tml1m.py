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


class TML1MDataset(Dataset):
    r"""TML1MDataset is a multi-table relational dataset containing 3 tables,
    as collected in the `rLLM: Relational Table Learning with LLMs
    <https://arxiv.org/abs/2407.20157>`__ paper.

    It includes three tables: users, movies and ratings tables.
    The users table includes information about users,
    such as gender and occupation.
    The movies table contains information about movies,
    such as duration and plot.
    The ratings table represents the interaction information
    between the user and movie tables.
    In addition, the embeddings of movies table using
    `all-MiniLM-L6-v2` model are also provided.
    The default task of this dataset is to predict user's age.

    Args:
        cached_dir (str): Root directory where dataset should be saved.
        forced_reload (bool): If set to `True`, this dataset will be
            re-process again.

    .. parsed-literal::

        Table1: users
        ---------------
            Statics:
            Name        Users     Features
            Size        6,040     5

        Table2: movies
        ------------------
            Statics:
            Name        Movies     Features
            nodes       3,883      11

        Table3: ratings
        ------------------
            Statics:
            Name        Ratings     Features
            nodes       1,000,209   4
    """

    url = "https://github.com/rllm-project/rllm_datasets/raw/refs/heads/main/sjtutables/tml1m.zip"  # noqa

    def __init__(
        self, cached_dir: str, force_reload: Optional[bool] = False, transform=None
    ) -> None:
        self.name = "tml1m"
        root = os.path.join(cached_dir, self.name)
        super().__init__(root, force_reload=force_reload)

        # Table_MovieLens1M data_list
        # 0: users_table
        # 1: movies_table
        # 2: ratings_table
        # 3: movie_embeddings
        self.data_list: List[TableData] = [
            TableData.load(self.processed_paths[0]),
            TableData.load(self.processed_paths[1]),
            TableData.load(self.processed_paths[2]),
            # TODO: Get this movie embedding from movie TableData
            torch.from_numpy(np.load(osp.join(self.raw_dir, "embeddings.npy"))),
        ]

        self.transform = transform
        if self.transform is not None:
            self.data_list[0] = self.transform(self.data_list[0])

    @property
    def raw_filenames(self):
        return ["users.csv", "movies.csv", "ratings.csv", "masks.pt"]

    @property
    def processed_filenames(self):
        return ["user_data.pt", "movie_data.pt", "rating_data.pt"]

    def process(self):
        r"""
        process data and save to './cached_dir/{dataset}/processed/'.
        """
        os.makedirs(self.processed_dir, exist_ok=True)
        # Users Data
        path = osp.join(self.raw_dir, self.raw_filenames[0])
        user_df = pd.read_csv(path, index_col=["UserID"])
        col_types = {
            "Gender": ColType.CATEGORICAL,
            "Age": ColType.CATEGORICAL,
            "Occupation": ColType.CATEGORICAL,
            "Zip-code": ColType.CATEGORICAL,
        }
        # Create masks
        masks_path = osp.join(self.raw_dir, self.raw_filenames[3])
        masks = torch.load(masks_path, weights_only=False)
        TableData(
            df=user_df,
            col_types=col_types,
            target_col="Age",
            train_mask=masks["train_mask"],
            val_mask=masks["val_mask"],
            test_mask=masks["test_mask"],
        ).save(self.processed_paths[0])

        # Movies Data
        path = osp.join(self.raw_dir, self.raw_filenames[1])
        movie_df = pd.read_csv(path, index_col=["MovieID"])
        # TODO: Use Text data in movies.csv to get embeddings.
        col_types = {
            "Year": ColType.NUMERICAL,
        }
        TableData(df=movie_df, col_types=col_types).save(self.processed_paths[1])

        # Ratings Data
        path = osp.join(self.raw_dir, self.raw_filenames[2])
        rating_df = pd.read_csv(path)
        col_types = {
            "UserID": ColType.NUMERICAL,
            "MovieID": ColType.NUMERICAL,
            "Rating": ColType.NUMERICAL,
        }
        TableData(df=rating_df, col_types=col_types).save(self.processed_paths[2])

    def download(self):
        os.makedirs(self.raw_dir, exist_ok=True)
        path = download_url(self.url, self.raw_dir, "TML1M.zip")
        extract_zip(path, self.raw_dir)
        os.remove(path)

    def __len__(self):
        return 3

    def __getitem__(self, index: int):
        if index < 0 or index > self.__len__():
            raise IndexError
        return self.data_list[index]
