from typing import Optional, List
import os
import os.path as osp

import pandas as pd
import torch

from rllm.types import ColType
from rllm.data.table_data import TableData
from rllm.datasets.dataset import Dataset
from rllm.utils.download import download_url
from rllm.utils.extract import extract_zip


class TLF2KDataset(Dataset):
    r"""TLF2KDataset is a multi-table relational dataset containing 3 tables,
    as collected in the `rLLM: Relational Table Learning with LLMs
    <https://arxiv.org/abs/2407.20157>`__ paper.

    It contains three tables: users, movies and ratings.
    The artists table includes information about artists,
    such as location and genre.
    The user_artists table contains the interaction between the user and artist
    as format: [user, artist, listening_count].
    The user_friends table represents bi-directional friendship between users.
    The default task of this dataset is to predict artists's genre.

    Args:
        cached_dir (str): Root directory where dataset should be saved.
        forced_reload (bool): If set to `True`, this dataset will be
            re-process again.

    .. parsed-literal::

        Table1: artists
        ---------------
            Statics:
            Name        Users     Features
            Size        9,047     10

        Table2: user_artists
        ------------------
            Statics:
            Name        Movies     Features
            nodes       80,009     3

        Table3: user_friends
        ------------------
            Statics:
            Name        Ratings     Features
            nodes       12,717      2
    """

    url = "https://raw.githubusercontent.com/rllm-project/rllm_datasets/refs/heads/main/sjtutables/tlf2k.zip"  # noqa

    def __init__(self, cached_dir: str, force_reload: Optional[bool] = False) -> None:
        self.name = "tlf2k"
        root = os.path.join(cached_dir, self.name)
        super().__init__(root, force_reload=force_reload)

        # Table_LastFM2K data_list:
        # 0: artists_table
        # 1: user_artists_table
        # 2: user_friends_table
        self.data_list: List[TableData] = [
            TableData.load(self.processed_paths[0]),
            TableData.load(self.processed_paths[1]),
            TableData.load(self.processed_paths[2]),
        ]

    @property
    def raw_filenames(self):
        return ["artists.csv", "user_artists.csv", "user_friends.csv", "masks.pt"]

    @property
    def processed_filenames(self):
        return ["artists_data.pt", "user_artists_data.pt", "user_friends_data.pt"]

    def process(self):
        r"""
        process data and save to './cached_dir/{dataset}/processed/'.
        """
        os.makedirs(self.processed_dir, exist_ok=True)
        # Artists Data
        path = osp.join(self.raw_dir, self.raw_filenames[0])
        artist_df = pd.read_csv(path)
        col_types = {
            # TODO: Process these feature with column type `Text`
            "type": ColType.CATEGORICAL,
            "name": ColType.CATEGORICAL,
            "born": ColType.CATEGORICAL,
            "yearsActive": ColType.CATEGORICAL,
            "location": ColType.CATEGORICAL,
            "biography": ColType.CATEGORICAL,
            "label": ColType.CATEGORICAL,
        }
        # Create masks
        masks_path = osp.join(self.raw_dir, self.raw_filenames[3])
        masks = torch.load(masks_path, weights_only=False)
        TableData(
            df=artist_df,
            col_types=col_types,
            target_col="label",
            train_mask=masks["train_mask"],
            val_mask=masks["val_mask"],
            test_mask=masks["test_mask"],
        ).save(self.processed_paths[0])

        # User-Artist Relationship
        path = osp.join(self.raw_dir, self.raw_filenames[1])
        ua_df = pd.read_csv(path)
        col_types = {
            "userID": ColType.NUMERICAL,
            "artistID": ColType.NUMERICAL,
        }
        TableData(df=ua_df, col_types=col_types).save(self.processed_paths[1])

        # User-user Relationship
        path = osp.join(self.raw_dir, self.raw_filenames[2])
        uu_df = pd.read_csv(path)
        col_types = {
            "userID": ColType.NUMERICAL,
            "friendID": ColType.NUMERICAL,
        }
        TableData(df=uu_df, col_types=col_types).save(self.processed_paths[2])

    def download(self):
        os.makedirs(self.raw_dir, exist_ok=True)
        path = download_url(self.url, self.raw_dir, "TLF2K.zip")
        extract_zip(path, self.raw_dir)
        os.remove(path)

    def __len__(self):
        return 3

    def __getitem__(self, index: int):
        if index < 0 or index > self.__len__():
            raise IndexError
        return self.data_list[index]
