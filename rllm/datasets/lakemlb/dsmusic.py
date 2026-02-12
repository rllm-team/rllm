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


class DSMusicDataset(Dataset):
    url = "https://github.com/FeiyuPan/LakeMLB_datasets/raw/refs/heads/main/datasets/dsmusic.zip"

    def __init__(self, cached_dir: str, csv_name: str = "discogs.csv", mask_name: str = "mask.pt",
                 force_reload: Optional[bool] = False, transform=None, device=None) -> None:
        self.name = "table_dsmusic"
        self.csv_name = csv_name
        self.mask_name = mask_name
        root = os.path.join(cached_dir, self.name)
        super().__init__(root, force_reload=force_reload)

        self.data_list: List[TableData] = [
            TableData.load(self.processed_paths[0])
        ]
        self.transform = transform
        if self.transform is not None:
            if device is not None:
                self.data_list[0] = self.transform(self.data_list[0]).to(device)
            else:
                self.data_list[0] = self.transform(self.data_list[0])

    @property
    def raw_filenames(self):
        return [
            "discogs.csv",
            "dsmusic_da.csv",
            "dsmusic_enriched.csv",
            "mask_da.pt",
            "mask.pt",
            "spotify.csv",
        ]

    @property
    def processed_filenames(self):
        base = osp.splitext(self.csv_name)[0]
        return [f"dsmusic_{base}_data.pt"]

    def process(self):
        os.makedirs(self.processed_dir, exist_ok=True)
        csv_path = osp.join(self.raw_dir, self.csv_name)
        masks_path = osp.join(self.raw_dir, self.mask_name)

        df = pd.read_csv(csv_path)

        col_types = {
            "title": ColType.CATEGORICAL,
            "release_year": ColType.NUMERICAL,
            "artists": ColType.CATEGORICAL,
            "genres": ColType.CATEGORICAL,
            "region": ColType.CATEGORICAL,
            # aux table cols
            # "track_id": ColType.CATEGORICAL,
            # "album_name": ColType.CATEGORICAL,
            # "popularity": ColType.NUMERICAL,
            # "duration_ms": ColType.NUMERICAL,
            # "explicit": ColType.CATEGORICAL,
            # "danceability": ColType.NUMERICAL,
            # "energy": ColType.NUMERICAL,
            # "key": ColType.NUMERICAL,
            # "loudness": ColType.NUMERICAL,
            # "mode": ColType.NUMERICAL,
            # "speechiness": ColType.NUMERICAL,
            # "acousticness": ColType.NUMERICAL,
            # "instrumentalness": ColType.NUMERICAL,
            # "liveness": ColType.NUMERICAL,
            # "valence": ColType.NUMERICAL,
            # "tempo": ColType.NUMERICAL,
            # "time_signature": ColType.NUMERICAL,
            # "track_genre": ColType.CATEGORICAL,
        }

        masks = torch.load(masks_path, weights_only=False)
        data = TableData(
            df=df,
            col_types=col_types,
            target_col="genres",
            train_mask=masks["train_mask"],
            val_mask=masks["val_mask"],
            test_mask=masks["test_mask"],
        )
        data.save(self.processed_paths[0])

    def download(self):
        os.makedirs(self.raw_dir, exist_ok=True)
        path = download_url(self.url, self.raw_dir, "dsmusic.zip")
        extract_zip(path, self.raw_dir)
        os.remove(path)

    def __len__(self):
        return 1

    def __getitem__(self, index: int):
        if index != 0:
            raise IndexError
        return self.data_list[0]
