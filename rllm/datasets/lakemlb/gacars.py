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
        csv_name (str): Name of the CSV file to use. Default is "german.csv".
        mask_name (str): Name of the mask file. Default is "mask_german.pt".
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

    Note:
        The columns commented out in col_types (under ``# aux table cols``) belong to
        the auxiliary table. They are commented here for convenience when running
        merged tables.
    """

    url = "https://github.com/FeiyuPan/LakeMLB_datasets/raw/refs/heads/main/datasets/gacars.zip"

    def __init__(self, cached_dir: str, csv_name: str = "german.csv", mask_name: str = "mask_german.pt",
                 force_reload: Optional[bool] = False, transform=None, device=None) -> None:
        self.name = "table_gacars"
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
            "australian_25pct.csv",
            "australian_50pct.csv",
            "australian_75pct.csv",
            "australian.csv",
            "gacars_da.csv",
            "gacars_fa.csv",
            "german.csv",
            "mask_australian_25pct.pt",
            "mask_australian_50pct.pt",
            "mask_australian_75pct.pt",
            "mask_australian.pt",
            "mask_da.pt",
            "mask_german.pt",
        ]

    @property
    def processed_filenames(self):
        base = osp.splitext(self.csv_name)[0]
        return [f"gacars_{base}_data.pt"]

    def process(self):
        os.makedirs(self.processed_dir, exist_ok=True)
        csv_path = osp.join(self.raw_dir, self.csv_name)
        masks_path = osp.join(self.raw_dir, self.mask_name)

        df_temp = pd.read_csv(csv_path, nrows=0, encoding='latin-1', low_memory=False)
        df_temp.columns = df_temp.columns.str.strip()

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
            # "Car/Suv": ColType.CATEGORICAL,
            # "Title": ColType.CATEGORICAL,
            # "UsedOrNew": ColType.CATEGORICAL,
            # "Transmission": ColType.CATEGORICAL,
            # "Engine": ColType.CATEGORICAL,
            # "DriveType": ColType.CATEGORICAL,
            # "FuelType": ColType.CATEGORICAL, #da
            # "FuelConsumption": ColType.CATEGORICAL,
            # "Kilometres": ColType.NUMERICAL,
            # "ColourExtInt": ColType.CATEGORICAL,
            # "Location": ColType.CATEGORICAL,
            # "CylindersinEngine": ColType.CATEGORICAL,
            # "BodyType": ColType.CATEGORICAL,
            # "Doors": ColType.CATEGORICAL,
            # "Seats": ColType.CATEGORICAL,
            # "Price": ColType.CATEGORICAL, #da
        }

        dtype_dict = {}
        for col_name, col_type in col_types.items():
            if col_type == ColType.CATEGORICAL and col_name in df_temp.columns:
                dtype_dict[col_name] = str

        df = pd.read_csv(csv_path, encoding='latin-1', low_memory=False, dtype=dtype_dict)
        df.columns = df.columns.str.strip()

        for col_name, col_type in col_types.items():
            if col_type == ColType.CATEGORICAL and col_name in df.columns:
                df[col_name] = df[col_name].astype(str)
                df[col_name] = df[col_name].replace(['nan', 'None', 'NaN', '<NA>'], 'Missing')
            elif col_type == ColType.NUMERICAL and col_name in df.columns:
                df[col_name] = df[col_name].fillna(0)

        masks = torch.load(masks_path, weights_only=False)
        data = TableData(
            df=df,
            col_types=col_types,
            target_col="price_in_euro",
            train_mask=masks["train_mask"],
            val_mask=masks["val_mask"],
            test_mask=masks["test_mask"],
        )
        data.save(self.processed_paths[0])

    def download(self):
        os.makedirs(self.raw_dir, exist_ok=True)
        path = download_url(self.url, self.raw_dir, "gacars.zip")
        extract_zip(path, self.raw_dir)
        os.remove(path)

    def __len__(self):
        return 1

    def __getitem__(self, index: int):
        if index != 0:
            raise IndexError
        return self.data_list[0]
