from typing import List, Optional
import os
import os.path as osp

import pandas as pd
import torch

from rllm.types import ColType
from rllm.data.table_data import TableData
from rllm.datasets.dataset import Dataset
from rllm.utils.download import download_url
from rllm.utils.extract import extract_zip


class NCTaxiDataset(Dataset):
    r"""A LakeMLB union-based dataset for taxi-trip classification.

    The task table contains New York taxi trips and the auxiliary table
    contains Chicago taxi trips. The prediction target is the New York
    drop-off location.

    Args:
        cached_dir (str): Root directory where the dataset is stored.
        force_reload (bool): Whether to process the raw files again.
        transform: Optional transform applied to each processed table.
        device: Optional device for transformed tables.
    """

    url = (
        "https://media.githubusercontent.com/media/zhengwang100/LakeMLB/"
        "main/benckmark/union_based/nctaxi.zip"
    )

    def __init__(
        self,
        cached_dir: str,
        force_reload: Optional[bool] = False,
        transform=None,
        device=None,
    ) -> None:
        self.name = "table_nctaxi"
        root = os.path.join(cached_dir, self.name)
        super().__init__(root, force_reload=force_reload)

        self.data_list: List[TableData] = [
            TableData.load(path) for path in self.processed_paths
        ]
        self.transform = transform
        if self.transform is not None:
            for i, data in enumerate(self.data_list):
                self.data_list[i] = (
                    self.transform(data).to(device)
                    if device is not None
                    else self.transform(data)
                )

    @property
    def raw_filenames(self):
        return [
            "newyork_taxi.csv",
            "chicago_taxi.csv",
            "nctaxi_da.csv",
            "nctaxi_fa.csv",
            "mask_newyork_taxi.pt",
            "mask_da.pt",
        ]

    @property
    def processed_filenames(self):
        return [
            "newyork_taxi_data.pt",
            "chicago_taxi_data.pt",
            "nctaxi_da_data.pt",
            "nctaxi_fa_data.pt",
        ]

    def process(self):
        os.makedirs(self.processed_dir, exist_ok=True)

        newyork_col_types = {
            "vendorid": ColType.CATEGORICAL,
            "tpep_pickup_datetime": ColType.CATEGORICAL,
            "tpep_dropoff_datetime": ColType.CATEGORICAL,
            "passenger_count": ColType.NUMERICAL,
            "trip_distance": ColType.NUMERICAL,
            "ratecodeid": ColType.CATEGORICAL,
            "store_and_fwd_flag": ColType.CATEGORICAL,
            "pulocationid": ColType.CATEGORICAL,
            "dolocationid": ColType.CATEGORICAL,
            "payment_type": ColType.CATEGORICAL,
            "fare_amount": ColType.NUMERICAL,
            "extra": ColType.NUMERICAL,
            "mta_tax": ColType.NUMERICAL,
            "tip_amount": ColType.NUMERICAL,
            "tolls_amount": ColType.NUMERICAL,
            "improvement_surcharge": ColType.NUMERICAL,
            "total_amount": ColType.NUMERICAL,
            "congestion_surcharge": ColType.NUMERICAL,
            "airport_fee": ColType.NUMERICAL,
        }
        chicago_col_types = {
            "trip_id": ColType.CATEGORICAL,
            "taxi_id": ColType.CATEGORICAL,
            "trip_start_timestamp": ColType.CATEGORICAL,
            "trip_end_timestamp": ColType.CATEGORICAL,
            "trip_seconds": ColType.NUMERICAL,
            "trip_miles": ColType.NUMERICAL,
            "pickup_census_tract": ColType.CATEGORICAL,
            "pickup_community_area": ColType.CATEGORICAL,
            "dropoff_community_area": ColType.CATEGORICAL,
            "fare": ColType.NUMERICAL,
            "tips": ColType.NUMERICAL,
            "tolls": ColType.NUMERICAL,
            "extras": ColType.NUMERICAL,
            "trip_total": ColType.NUMERICAL,
            "payment_type": ColType.CATEGORICAL,
            "company": ColType.CATEGORICAL,
            "pickup_centroid_latitude": ColType.NUMERICAL,
            "pickup_centroid_longitude": ColType.NUMERICAL,
            "pickup_centroid_location": ColType.CATEGORICAL,
        }

        newyork_df = pd.read_csv(self.raw_paths[0], low_memory=False)
        masks = torch.load(self.raw_paths[4], weights_only=False)
        TableData(
            df=newyork_df,
            col_types=newyork_col_types,
            target_col="dolocationid",
            train_mask=masks["train_mask"],
            val_mask=masks["val_mask"],
            test_mask=masks["test_mask"],
        ).save(self.processed_paths[0])

        chicago_df = pd.read_csv(self.raw_paths[1], low_memory=False)
        TableData(
            df=chicago_df,
            col_types=chicago_col_types,
            target_col="dropoff_community_area",
        ).save(self.processed_paths[1])

        da_col_types = {
            **newyork_col_types,
            "trip_id": ColType.CATEGORICAL,
            "taxi_id": ColType.CATEGORICAL,
            "trip_start_timestamp": ColType.CATEGORICAL,
            "trip_end_timestamp": ColType.CATEGORICAL,
            "trip_seconds": ColType.NUMERICAL,
            "trip_miles": ColType.NUMERICAL,
            "pickup_census_tract": ColType.CATEGORICAL,
            "pickup_community_area": ColType.CATEGORICAL,
            "fare": ColType.NUMERICAL,
            "tips": ColType.NUMERICAL,
            "tolls": ColType.NUMERICAL,
            "extras": ColType.NUMERICAL,
            "trip_total": ColType.NUMERICAL,
            "company": ColType.CATEGORICAL,
            "pickup_centroid_latitude": ColType.NUMERICAL,
            "pickup_centroid_longitude": ColType.NUMERICAL,
            "pickup_centroid_location": ColType.CATEGORICAL,
        }
        nctaxi_da_df = pd.read_csv(self.raw_paths[2], low_memory=False)
        masks = torch.load(self.raw_paths[5], weights_only=False)
        TableData(
            df=nctaxi_da_df,
            col_types=da_col_types,
            target_col="dolocationid",
            train_mask=masks["train_mask"],
            val_mask=masks["val_mask"],
            test_mask=masks["test_mask"],
        ).save(self.processed_paths[2])

        fa_col_types = {
            **da_col_types,
            "dropoff_community_area": ColType.CATEGORICAL,
            "aux_payment_type": ColType.CATEGORICAL,
        }
        nctaxi_fa_df = pd.read_csv(self.raw_paths[3], low_memory=False)
        masks = torch.load(self.raw_paths[4], weights_only=False)
        TableData(
            df=nctaxi_fa_df,
            col_types=fa_col_types,
            target_col="dolocationid",
            train_mask=masks["train_mask"],
            val_mask=masks["val_mask"],
            test_mask=masks["test_mask"],
        ).save(self.processed_paths[3])

    def download(self):
        os.makedirs(self.raw_dir, exist_ok=True)
        path = download_url(self.url, self.raw_dir, "nctaxi.zip")
        extract_zip(path, self.raw_dir)
        os.remove(path)

    def __len__(self):
        return 4

    def __getitem__(self, index: int):
        if index < 0 or index >= len(self.data_list):
            raise IndexError
        return self.data_list[index]
