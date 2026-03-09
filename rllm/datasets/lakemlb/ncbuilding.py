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
from rllm.utils.csv_utils import read_csv_with_fallback_encodings


class NCBuildingDataset(Dataset):
    r"""NCBuildingDataset is a tabular dataset designed for weakly related (Union-based)
    table scenarios in Data Lake(House) settings, as collected in the `LakeMLB: Data Lake
    Machine Learning Benchmark <https://arxiv.org/abs/2602.10441>`__ paper.

    The dataset focuses on building violation complaint incidents and comprises two weakly
    related tables: a task table (New York complaint reports) and an auxiliary table
    (Chicago complaint reports). The task table contains building violation complaint
    records from January 2023 to December 2024. The auxiliary table contains building
    violation complaint records from January 2023 to December 2024. The two tables exhibit
    a weak association (Union relationship), where information from the auxiliary table
    can be leveraged to enhance machine learning performance on the task table. The default
    task is to predict the type of building violation being complained about (a classification
    task).

    Args:
        cached_dir (str): Root directory where dataset should be saved.
        force_reload (bool): If set to `True`, this dataset will be re-process again.
        transform: Optional transform to be applied on the data.
        device: Optional device to move the transformed data to.

    .. parsed-literal::

        Table1: newyork
        ---------------
            Statics:
            Name        Records     Features
            Size        30,000      40

        Table2: chicago
        ------------------
            Statics:
            Name        Records     Features
            Size        37,000      23

    """

    url = "https://raw.githubusercontent.com/zhengwang100/LakeMLB/main/benckmark/union_based/ncbuilding.zip"

    def __init__(self, cached_dir: str, force_reload: Optional[bool] = False, transform=None, device=None
    ) -> None:
        self.name = "table_ncbuilding"
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
            "newyork.csv",
            "chicago.csv",
            "ncbuilding_da.csv",
            "ncbuilding_fa.csv",
            "mask_newyork.pt",
            "mask_da.pt",
        ]

    @property
    def processed_filenames(self):
        return [
            "newyork_data.pt",
            "chicago_data.pt",
            "ncbuilding_da_data.pt",
            "ncbuilding_fa_data.pt",
        ]

    def process(self):
        os.makedirs(self.processed_dir, exist_ok=True)
        # New York Data
        csv_path = osp.join(self.raw_dir, self.raw_filenames[0])
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
        }
        newyork_df = pd.read_csv(csv_path, low_memory=False)
        masks_path = osp.join(self.raw_dir, self.raw_filenames[4])
        masks = torch.load(masks_path, weights_only=False)
        TableData(
            df=newyork_df,
            col_types=col_types,
            target_col="StatuteCodes",
            train_mask=masks["train_mask"],
            val_mask=masks["val_mask"],
            test_mask=masks["test_mask"],
        ).save(self.processed_paths[0])

        # Chicago Data
        csv_path = osp.join(self.raw_dir, self.raw_filenames[1])
        col_types = {
            "ID": ColType.NUMERICAL,
            "VIOLATION LAST MODIFIED DATE": ColType.CATEGORICAL,
            "VIOLATION DATE": ColType.CATEGORICAL,
            "VIOLATION STATUS":  ColType.CATEGORICAL,
            "VIOLATION STATUS DATE": ColType.CATEGORICAL,
            "VIOLATION DESCRIPTION":  ColType.CATEGORICAL,
            "VIOLATION LOCATION": ColType.CATEGORICAL,
            "VIOLATION INSPECTOR COMMENTS": ColType.CATEGORICAL,
            "INSPECTOR ID": ColType.NUMERICAL,
            "INSPECTION NUMBER": ColType.NUMERICAL,
            "INSPECTION STATUS": ColType.CATEGORICAL,
            "INSPECTION WAIVED": ColType.CATEGORICAL,
            "INSPECTION CATEGORY": ColType.CATEGORICAL,
            "ADDRESS": ColType.CATEGORICAL,
            "STREET NUMBER": ColType.NUMERICAL,
            "STREET DIRECTION": ColType.CATEGORICAL,
            "STREET NAME": ColType.CATEGORICAL,
            "STREET TYPE": ColType.CATEGORICAL,
            "PROPERTY GROUP": ColType.NUMERICAL,
            "SSA": ColType.NUMERICAL,
            "LATITUDE": ColType.NUMERICAL,
            "LONGITUDE": ColType.NUMERICAL,
            "LOCATION": ColType.CATEGORICAL,
        }
        chicago_df = pd.read_csv(csv_path, low_memory=False)
        TableData(
            df=chicago_df,
            col_types=col_types,
            target_col="VIOLATION DESCRIPTION",
        ).save(self.processed_paths[1])

        # Merged Data(DA)
        csv_path = osp.join(self.raw_dir, self.raw_filenames[2])
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
            "ID": ColType.NUMERICAL,
            "VIOLATION LAST MODIFIED DATE": ColType.CATEGORICAL,
            "VIOLATION DATE": ColType.CATEGORICAL,
            # "VIOLATION STATUS":  ColType.CATEGORICAL, #da
            "VIOLATION STATUS DATE": ColType.CATEGORICAL,
            # "VIOLATION DESCRIPTION":  ColType.CATEGORICAL, #da
            "VIOLATION LOCATION": ColType.CATEGORICAL,
            "VIOLATION INSPECTOR COMMENTS": ColType.CATEGORICAL,
            "INSPECTOR ID": ColType.NUMERICAL,
            "INSPECTION NUMBER": ColType.NUMERICAL,
            "INSPECTION STATUS": ColType.CATEGORICAL,
            "INSPECTION WAIVED": ColType.CATEGORICAL,
            "INSPECTION CATEGORY": ColType.CATEGORICAL,
            "ADDRESS": ColType.CATEGORICAL,
            "STREET NUMBER": ColType.NUMERICAL,
            "STREET DIRECTION": ColType.CATEGORICAL,
            # "STREET NAME": ColType.CATEGORICAL, #da
            "STREET TYPE": ColType.CATEGORICAL,
            "PROPERTY GROUP": ColType.NUMERICAL,
            "SSA": ColType.NUMERICAL,
            # "LATITUDE": ColType.NUMERICAL, #da
            # "LONGITUDE": ColType.NUMERICAL, #da
            "LOCATION": ColType.CATEGORICAL,
        }
        ncbuilding_da_df = pd.read_csv(csv_path, low_memory=False)
        masks_path = osp.join(self.raw_dir, self.raw_filenames[5])
        masks = torch.load(masks_path, weights_only=False)
        TableData(
            df=ncbuilding_da_df,
            col_types=col_types,
            target_col="StatuteCodes",
            train_mask=masks["train_mask"],
            val_mask=masks["val_mask"],
            test_mask=masks["test_mask"],
        ).save(self.processed_paths[2])

        # Merged Data(FA)
        csv_path = osp.join(self.raw_dir, self.raw_filenames[3])
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
            "ID": ColType.NUMERICAL,
            "VIOLATION LAST MODIFIED DATE": ColType.CATEGORICAL,
            "VIOLATION DATE": ColType.CATEGORICAL,
            "VIOLATION STATUS":  ColType.CATEGORICAL,
            "VIOLATION STATUS DATE": ColType.CATEGORICAL,
            "VIOLATION DESCRIPTION":  ColType.CATEGORICAL,
            "VIOLATION LOCATION": ColType.CATEGORICAL,
            "VIOLATION INSPECTOR COMMENTS": ColType.CATEGORICAL,
            "INSPECTOR ID": ColType.NUMERICAL,
            "INSPECTION NUMBER": ColType.NUMERICAL,
            "INSPECTION STATUS": ColType.CATEGORICAL,
            "INSPECTION WAIVED": ColType.CATEGORICAL,
            "INSPECTION CATEGORY": ColType.CATEGORICAL,
            "ADDRESS": ColType.CATEGORICAL,
            "STREET NUMBER": ColType.NUMERICAL,
            "STREET DIRECTION": ColType.CATEGORICAL,
            "STREET NAME": ColType.CATEGORICAL,
            "STREET TYPE": ColType.CATEGORICAL,
            "PROPERTY GROUP": ColType.NUMERICAL,
            "SSA": ColType.NUMERICAL,
            "LATITUDE": ColType.NUMERICAL,
            "LONGITUDE": ColType.NUMERICAL,
            "LOCATION": ColType.CATEGORICAL,
        }
        ncbuilding_fa_df = pd.read_csv(csv_path, low_memory=False)
        masks_path = osp.join(self.raw_dir, self.raw_filenames[4])
        masks = torch.load(masks_path, weights_only=False)
        TableData(
            df=ncbuilding_fa_df,
            col_types=col_types,
            target_col="StatuteCodes",
            train_mask=masks["train_mask"],
            val_mask=masks["val_mask"],
            test_mask=masks["test_mask"],
        ).save(self.processed_paths[3])

    def download(self):
        os.makedirs(self.raw_dir, exist_ok=True)
        path = download_url(self.url, self.raw_dir, "ncbuilding.zip")
        extract_zip(path, self.raw_dir)
        os.remove(path)

    def __len__(self):
        return 4

    def __getitem__(self, index: int):
        if index < 0 or index >= len(self.data_list):
            raise IndexError
        return self.data_list[index]
