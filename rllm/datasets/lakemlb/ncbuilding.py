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


class NCBuildingDataset(Dataset):
    url = "https://github.com/FeiyuPan/LakeMLB_datasets/raw/refs/heads/main/datasets/ncbuilding.zip"

    def __init__(self, cached_dir: str, csv_name: str = "newyork.csv", mask_name: str = "mask_newyork.pt",
                 force_reload: Optional[bool] = False, transform=None, device=None) -> None:
        self.name = "table_ncbuilding"
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
            "chicago_25pct.csv",
            "chicago_50pct.csv",
            "chicago_75pct.csv",
            "chicago.csv",
            "mask_chicago_25pct.pt",
            "mask_chicago_50pct.pt",
            "mask_chicago_75pct.pt",
            "mask_chicago.pt",
            "mask_da.pt",
            "mask_newyork.pt",
            "ncbuilding_da.csv",
            "ncbuilding_fa.csv",
            "newyork.csv",
        ]

    @property
    def processed_filenames(self):
        base = osp.splitext(self.csv_name)[0]
        return [f"ncbuilding_{base}_data.pt"]

    def process(self):
        os.makedirs(self.processed_dir, exist_ok=True)
        csv_path = osp.join(self.raw_dir, self.csv_name)
        masks_path = osp.join(self.raw_dir, self.mask_name)
        df = None
        for encoding in ['utf-8', 'utf-8-sig', 'gbk', 'gb18030', 'latin1']:
            try:
                df = pd.read_csv(csv_path, encoding=encoding)
                print(f"Successfully loaded CSV with encoding: {encoding}")
                print(f"Columns found: {list(df.columns[:5])}...")
                break
            except (UnicodeDecodeError, Exception) as e:
                print(f"Failed to load with encoding {encoding}: {e}")
                continue

        if df is None:
            raise ValueError("Failed to load CSV with any supported encoding")

        col_types = {
            "ViolationID": ColType.NUMERICAL,
            "BuildingID": ColType.NUMERICAL,
            "RegistrationID": ColType.NUMERICAL,
            "BoroID": ColType.CATEGORICAL,
            "Borough": ColType.CATEGORICAL,
            "HouseNumber": ColType.CATEGORICAL,
            "LowHouseNumber": ColType.CATEGORICAL,
            "HighHouseNumber": ColType.CATEGORICAL,
            "StreetName": ColType.CATEGORICAL,
            "StreetCode": ColType.CATEGORICAL,
            "Postcode": ColType.CATEGORICAL,
            "Apartment": ColType.CATEGORICAL,
            "Story": ColType.CATEGORICAL,
            "Block": ColType.CATEGORICAL,
            "Lot": ColType.CATEGORICAL,
            "Class": ColType.CATEGORICAL,
            "InspectionDate": ColType.CATEGORICAL,
            "ApprovedDate": ColType.CATEGORICAL,
            "OriginalCertifyByDate": ColType.CATEGORICAL,
            "OriginalCorrectByDate": ColType.CATEGORICAL,
            "NewCertifyByDate": ColType.CATEGORICAL,
            "NewCorrectByDate": ColType.CATEGORICAL,
            "CertifiedDate": ColType.CATEGORICAL,
            "NOVID": ColType.NUMERICAL,
            "StatuteCodes": ColType.CATEGORICAL,
            "NOVIssuedDate": ColType.CATEGORICAL,
            "CurrentStatusID": ColType.CATEGORICAL,
            "CurrentStatus": ColType.CATEGORICAL,
            "CurrentStatusDate": ColType.CATEGORICAL,
            "NovType": ColType.CATEGORICAL,
            "ViolationStatus": ColType.CATEGORICAL,
            "RentImpairing": ColType.CATEGORICAL,
            "Latitude": ColType.NUMERICAL,
            "Longitude": ColType.NUMERICAL,
            "CommunityBoard": ColType.CATEGORICAL,
            "CouncilDistrict": ColType.CATEGORICAL,
            "CensusTract": ColType.CATEGORICAL,
            "BIN": ColType.NUMERICAL,
            "BBL": ColType.NUMERICAL,
            "NTA": ColType.CATEGORICAL,
            # aux table cols
            # "ID": ColType.NUMERICAL,
            # "VIOLATION LAST MODIFIED DATE": ColType.CATEGORICAL,
            # "VIOLATION DATE": ColType.CATEGORICAL,
            # # "VIOLATION STATUS":  ColType.CATEGORICAL, #da
            # "VIOLATION STATUS DATE": ColType.CATEGORICAL,
            # # "VIOLATION DESCRIPTION":  ColType.CATEGORICAL, #da
            # "VIOLATION LOCATION": ColType.CATEGORICAL,
            # "VIOLATION INSPECTOR COMMENTS": ColType.CATEGORICAL,
            # "INSPECTOR ID": ColType.NUMERICAL,
            # "INSPECTION NUMBER": ColType.NUMERICAL,
            # "INSPECTION STATUS": ColType.CATEGORICAL,
            # "INSPECTION WAIVED": ColType.CATEGORICAL,
            # "INSPECTION CATEGORY": ColType.CATEGORICAL,
            # "ADDRESS": ColType.CATEGORICAL,
            # "STREET NUMBER": ColType.NUMERICAL,
            # "STREET DIRECTION": ColType.CATEGORICAL,
            # # "STREET NAME": ColType.CATEGORICAL, #da
            # "STREET TYPE": ColType.CATEGORICAL,
            # "PROPERTY GROUP": ColType.NUMERICAL,
            # "SSA": ColType.NUMERICAL,
            # # "LATITUDE": ColType.NUMERICAL, #da
            # # "LONGITUDE": ColType.NUMERICAL, #da
            # "LOCATION": ColType.CATEGORICAL,
        }

        masks = torch.load(masks_path, weights_only=False)
        data = TableData(
            df=df,
            col_types=col_types,
            target_col="StatuteCodes",
            train_mask=masks["train_mask"],
            val_mask=masks["val_mask"],
            test_mask=masks["test_mask"],
        )
        data.save(self.processed_paths[0])

    def download(self):
        os.makedirs(self.raw_dir, exist_ok=True)
        path = download_url(self.url, self.raw_dir, "ncbuilding.zip")
        extract_zip(path, self.raw_dir)
        os.remove(path)

    def __len__(self):
        return 1

    def __getitem__(self, index: int):
        if index != 0:
            raise IndexError
        return self.data_list[0]
