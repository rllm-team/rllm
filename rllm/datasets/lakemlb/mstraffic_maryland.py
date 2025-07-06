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

class MSTrafficMarylandDataset(Dataset):
    r"""The MS-Traffic dataset for collision type prediction.
    
    This dataset contains traffic crash records for Maryland. The task is to
    predict the 'Collision Type'.

    The raw data is downloaded as a zip file and contains multiple files.
    This class processes 'Maryland.csv' and uses masks from 'T1_mask.pt'.
    """
    
    #url = "https://github.com/FeiyuPan/LakeMLB_datasets/raw/refs/heads/main/datasets/Crash20K.zip"
    
    archive_filename = "Crash20K.zip"

    def __init__(self, cached_dir: str, force_reload: Optional[bool] = False, transform=None, device=None) -> None:
        self.name = "mstraffic-maryland"
        root = os.path.join(cached_dir, self.name)
        super().__init__(root, force_reload=force_reload)

        self.data_list: List[TableData] = [
            TableData.load(self.processed_paths[0])
        ]
        
        self.transform = transform
        if self.transform is not None:
            data = self.data_list[0]
            data = self.transform(data)
            if device is not None:
                data = data.to(device)
            self.data_list[0] = data

    @property
    def raw_filenames(self) -> List[str]:
        return ["maryland.csv", "T1_mask.pt"]

    @property
    def processed_filenames(self) -> List[str]:
        return ["mstraffic_maryland_data.pt"]

    def process(self):
        """Processes the raw csv and mask files into a single TableData object."""
        os.makedirs(self.processed_dir, exist_ok=True)

        csv_path = osp.join(self.raw_dir, self.raw_filenames[0])
        masks_path = osp.join(self.raw_dir, self.raw_filenames[1])

        df = pd.read_csv(csv_path)
        
        col_types = {
            "Report Number": ColType.CATEGORICAL,
            "Local Case Number": ColType.CATEGORICAL,
            "Agency Name": ColType.CATEGORICAL,
            "ACRS Report Type": ColType.CATEGORICAL,
            "Crash Date/Time": ColType.CATEGORICAL,
            "Hit/Run": ColType.CATEGORICAL,
            "Route Type": ColType.CATEGORICAL,
            "Lane Direction": ColType.CATEGORICAL,
            "Lane Type": ColType.CATEGORICAL,
            "Number of Lanes": ColType.CATEGORICAL,
            "Direction": ColType.CATEGORICAL,
            "Distance": ColType.NUMERICAL,
            "Distance Unit": ColType.CATEGORICAL,
            "Road Grade": ColType.CATEGORICAL,
            "Road Name": ColType.CATEGORICAL,
            "Cross-Street Name": ColType.CATEGORICAL,
            "Off-Road Description": ColType.CATEGORICAL,
            "Related Non-Motorist": ColType.CATEGORICAL,
            "At Fault": ColType.CATEGORICAL,
            "Collision Type": ColType.CATEGORICAL,  # This is the target
            "Weather": ColType.CATEGORICAL,
            "Surface Condition": ColType.CATEGORICAL,
            "Light": ColType.CATEGORICAL,
            "Traffic Control": ColType.CATEGORICAL,
            "Driver Substance Abuse": ColType.CATEGORICAL,
            "Non-Motorist Substance Abuse": ColType.CATEGORICAL,
            "First Harmful Event": ColType.CATEGORICAL,
            "Second Harmful Event": ColType.CATEGORICAL,
            "Junction": ColType.CATEGORICAL,
            "Intersection Type": ColType.CATEGORICAL,
            "Road Alignment": ColType.CATEGORICAL,
            "Road Condition": ColType.CATEGORICAL,
            "Road Division": ColType.CATEGORICAL,
            "Latitude": ColType.NUMERICAL,
            "Longitude": ColType.NUMERICAL,
            "Location": ColType.CATEGORICAL,
        }
        
        masks = torch.load(masks_path)
        
        data = TableData(
            df=df,
            col_types=col_types,
            target_col="Collision Type",
            train_mask=masks["train_mask"],
            val_mask=masks["val_mask"],
            test_mask=masks["test_mask"],
        )
        data.save(self.processed_paths[0])

    def download(self):
        """Downloads the zip archive, extracts it, and removes the archive."""
        os.makedirs(self.raw_dir, exist_ok=True)
        path = download_url(self.url, self.raw_dir, self.archive_filename)
        extract_zip(path, self.raw_dir)
        os.remove(path)

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, index: int) -> TableData:
        if index >= len(self.data_list):
            raise IndexError
        return self.data_list[index]