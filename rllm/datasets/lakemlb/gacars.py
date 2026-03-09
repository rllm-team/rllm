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


class GACarsDataset(Dataset):
    r"""GACarsDataset is a tabular dataset designed for weakly related (Union-based)
    table scenarios in Data Lake(House) settings, as collected in the `LakeMLB: Data Lake
    Machine Learning Benchmark <https://arxiv.org/abs/2602.10441>`__ paper.

    The dataset focuses on used car sales transactions and comprises two weakly related
    tables: a task table (German used car listings) and an auxiliary table (Australian
    used car listings). The task table contains used car listing records collected in 2023.
    The auxiliary table contains used car listing records collected in 2023.
    The two tables exhibit a weak association (Union relationship), where information
    from the auxiliary table can be leveraged to enhance machine learning performance
    on the task table. The default task is to predict the price range of used car sales
    (a classification task).

    Args:
        cached_dir (str): Root directory where dataset should be saved.
        force_reload (bool): If set to `True`, this dataset will be re-process again.
        transform: Optional transform to be applied on the data.
        device: Optional device to move the transformed data to.

    .. parsed-literal::

        Table1: german
        ---------------
            Statics:
            Name        Records     Features
            Size        13,000      15

        Table2: australian
        ------------------
            Statics:
            Name        Records     Features
            Size        3,600       19

    """

    url = "https://raw.githubusercontent.com/zhengwang100/LakeMLB/main/benckmark/union_based/gacars.zip"

    def __init__(self, cached_dir: str, force_reload: Optional[bool] = False, transform=None, device=None
    ) -> None:
        self.name = "table_gacars"
        root = os.path.join(cached_dir, self.name)
        super().__init__(root, force_reload=force_reload)

        self.data_list: List[TableData] = [
            TableData.load(self.processed_paths[0]),
            TableData.load(self.processed_paths[1]),
            TableData.load(self.processed_paths[2]),
            TableData.load(self.processed_paths[3]),
        ]
        self.transform = transform
        if self.transform is not None:
            for i, data in enumerate(self.data_list):
                self.data_list[i] = (
                    self.transform(data).to(device) if device is not None
                    else self.transform(data)
                )

    @property
    def raw_filenames(self):
        return [
            "german.csv",
            "australian.csv",
            "gacars_da.csv",
            "gacars_fa.csv",
            "mask_german.pt",
            "mask_da.pt",
        ]

    @property
    def processed_filenames(self):
        return [
            "german_data.pt",
            "australian_data.pt",
            "gacars_da_data.pt",
            "gacars_fa_data.pt",
        ]

    def process(self):
        os.makedirs(self.processed_dir, exist_ok=True)
        # German Data
        csv_path = osp.join(self.raw_dir, self.raw_filenames[0])
        col_types = {
            "id": ColType.NUMERICAL,
            "brand": ColType.CATEGORICAL,
            "model": ColType.CATEGORICAL,
            "color": ColType.CATEGORICAL,
            "registration_date": ColType.CATEGORICAL,
            "year": ColType.NUMERICAL,
            "price_in_euro": ColType.CATEGORICAL,
            "power_kw": ColType.NUMERICAL,
            "power_ps": ColType.NUMERICAL,
            "transmission_type": ColType.CATEGORICAL,
            "fuel_type": ColType.CATEGORICAL,
            "fuel_consumption_l_100km": ColType.CATEGORICAL,
            "fuel_consumption_g_km": ColType.CATEGORICAL,
            "mileage_in_km": ColType.NUMERICAL,
            "offer_description": ColType.CATEGORICAL,
        }
        german_df = pd.read_csv(csv_path, low_memory=False)
        masks_path = osp.join(self.raw_dir, self.raw_filenames[4])
        masks = torch.load(masks_path, weights_only=False)
        TableData(
            df=german_df,
            col_types=col_types,
            target_col="price_in_euro",
            train_mask=masks["train_mask"],
            val_mask=masks["val_mask"],
            test_mask=masks["test_mask"],
        ).save(self.processed_paths[0])

        # Australian Data
        csv_path = osp.join(self.raw_dir, self.raw_filenames[1])
        col_types = {
            "Brand": ColType.CATEGORICAL,
            "Year": ColType.NUMERICAL,
            "Model": ColType.CATEGORICAL,
            "Car/Suv": ColType.CATEGORICAL,
            "Title": ColType.CATEGORICAL,
            "UsedOrNew": ColType.CATEGORICAL,
            "Transmission": ColType.CATEGORICAL,
            "Engine": ColType.CATEGORICAL,
            "DriveType": ColType.CATEGORICAL,
            "FuelType": ColType.CATEGORICAL,
            "FuelConsumption": ColType.CATEGORICAL,
            "Kilometres": ColType.NUMERICAL,
            "ColourExtInt": ColType.CATEGORICAL,
            "Location": ColType.CATEGORICAL,
            "CylindersinEngine": ColType.CATEGORICAL,
            "BodyType": ColType.CATEGORICAL,
            "Doors": ColType.CATEGORICAL,
            "Seats": ColType.CATEGORICAL,
            "Price": ColType.CATEGORICAL,
        }
        australian_df = pd.read_csv(csv_path, low_memory=False)
        TableData(
            df=australian_df,
            col_types=col_types,
            target_col="Price",
        ).save(self.processed_paths[1])

        # Merged Data(DA)
        csv_path = osp.join(self.raw_dir, self.raw_filenames[2])
        col_types = {
            "id": ColType.NUMERICAL,
            "brand": ColType.CATEGORICAL,
            "model": ColType.CATEGORICAL,
            "color": ColType.CATEGORICAL,
            "registration_date": ColType.CATEGORICAL,
            "year": ColType.NUMERICAL,
            "price_in_euro": ColType.CATEGORICAL,
            "power_kw": ColType.NUMERICAL,
            "power_ps": ColType.NUMERICAL,
            "transmission_type": ColType.CATEGORICAL,
            "fuel_type": ColType.CATEGORICAL,
            "fuel_consumption_l_100km": ColType.CATEGORICAL,
            "fuel_consumption_g_km": ColType.CATEGORICAL,
            "mileage_in_km": ColType.NUMERICAL,
            "offer_description": ColType.CATEGORICAL,
            # aux table cols
            # "Brand": ColType.CATEGORICAL, #da
            # "Year": ColType.NUMERICAL, #da
            # "Model": ColType.CATEGORICAL, #da
            "Car/Suv": ColType.CATEGORICAL,
            "Title": ColType.CATEGORICAL,
            "UsedOrNew": ColType.CATEGORICAL,
            "Transmission": ColType.CATEGORICAL,
            "Engine": ColType.CATEGORICAL,
            "DriveType": ColType.CATEGORICAL,
            # "FuelType": ColType.CATEGORICAL, #da
            "FuelConsumption": ColType.CATEGORICAL,
            "Kilometres": ColType.NUMERICAL,
            "ColourExtInt": ColType.CATEGORICAL,
            "Location": ColType.CATEGORICAL,
            "CylindersinEngine": ColType.CATEGORICAL,
            "BodyType": ColType.CATEGORICAL,
            "Doors": ColType.CATEGORICAL,
            "Seats": ColType.CATEGORICAL,
            # "Price": ColType.CATEGORICAL, #da
        }
        gacars_da_df = pd.read_csv(csv_path, low_memory=False)
        masks_path = osp.join(self.raw_dir, self.raw_filenames[5])
        masks = torch.load(masks_path, weights_only=False)
        TableData(
            df=gacars_da_df,
            col_types=col_types,
            target_col="price_in_euro",
            train_mask=masks["train_mask"],
            val_mask=masks["val_mask"],
            test_mask=masks["test_mask"],
        ).save(self.processed_paths[2])

        # Merged Data(FA)
        csv_path = osp.join(self.raw_dir, self.raw_filenames[3])
        col_types = {
            "id": ColType.NUMERICAL,
            "brand": ColType.CATEGORICAL,
            "model": ColType.CATEGORICAL,
            "color": ColType.CATEGORICAL,
            "registration_date": ColType.CATEGORICAL,
            "year": ColType.NUMERICAL,
            "price_in_euro": ColType.CATEGORICAL,
            "power_kw": ColType.NUMERICAL,
            "power_ps": ColType.NUMERICAL,
            "transmission_type": ColType.CATEGORICAL,
            "fuel_type": ColType.CATEGORICAL,
            "fuel_consumption_l_100km": ColType.CATEGORICAL,
            "fuel_consumption_g_km": ColType.CATEGORICAL,
            "mileage_in_km": ColType.NUMERICAL,
            "offer_description": ColType.CATEGORICAL,
            # aux table cols
            "Brand": ColType.CATEGORICAL,
            "Year": ColType.NUMERICAL,
            "Model": ColType.CATEGORICAL,
            "Car/Suv": ColType.CATEGORICAL,
            "Title": ColType.CATEGORICAL,
            "UsedOrNew": ColType.CATEGORICAL,
            "Transmission": ColType.CATEGORICAL,
            "Engine": ColType.CATEGORICAL,
            "DriveType": ColType.CATEGORICAL,
            "FuelType": ColType.CATEGORICAL,
            "FuelConsumption": ColType.CATEGORICAL,
            "Kilometres": ColType.NUMERICAL,
            "ColourExtInt": ColType.CATEGORICAL,
            "Location": ColType.CATEGORICAL,
            "CylindersinEngine": ColType.CATEGORICAL,
            "BodyType": ColType.CATEGORICAL,
            "Doors": ColType.CATEGORICAL,
            "Seats": ColType.CATEGORICAL,
            "Price": ColType.CATEGORICAL,
        }
        gacars_fa_df = pd.read_csv(csv_path, low_memory=False)
        masks_path = osp.join(self.raw_dir, self.raw_filenames[4])
        masks = torch.load(masks_path, weights_only=False)
        TableData(
            df=gacars_fa_df,
            col_types=col_types,
            target_col="price_in_euro",
            train_mask=masks["train_mask"],
            val_mask=masks["val_mask"],
            test_mask=masks["test_mask"],
        ).save(self.processed_paths[3])

    def download(self):
        os.makedirs(self.raw_dir, exist_ok=True)
        path = download_url(self.url, self.raw_dir, "gacars.zip")
        extract_zip(path, self.raw_dir)
        os.remove(path)

    def __len__(self):
        return 4

    def __getitem__(self, index: int):
        if index < 0 or index >= len(self.data_list):
            raise IndexError
        return self.data_list[index]
