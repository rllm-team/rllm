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


class MSTrafficDataset(Dataset):
    r"""MSTrafficDataset is a tabular dataset designed for weakly related (Union-based)
    table scenarios in Data Lake(House) settings, as collected in the `LakeMLB: Data Lake
    Machine Learning Benchmark <https://arxiv.org/abs/2602.10441>`__ paper.

    The dataset focuses on traffic collision incidents and comprises two weakly related
    tables: a task table (Maryland collision reports) and an auxiliary table (Seattle
    collision reports). The task table contains traffic collision records from January
    2017 to December 2023. The auxiliary table contains traffic collision records from
    January 2014 to December 2023. The two tables exhibit a weak association (Union
    relationship), where information from the auxiliary table can be leveraged to enhance
    machine learning performance on the task table. The default task is to predict the
    collision type of traffic incidents (a classification task).

    Args:
        cached_dir (str): Root directory where dataset should be saved.
        csv_name (str): Name of the CSV file to use. Default is "maryland.csv".
        mask_name (str): Name of the mask file. Default is "maryland_mask.pt".
        force_reload (bool): If set to `True`, this dataset will be re-process again.
        transform: Optional transform to be applied on the data.
        device: Optional device to move the transformed data to.

    .. parsed-literal::

        Table1: maryland
        ---------------
            Statics:
            Name        Records     Features
            Size        10,800      37

        Table2: seattle
        ------------------
            Statics:
            Name        Records     Features
            Size        10,800      50

    Note:
        The columns commented out in col_types (under ``# aux table cols``) belong to
        the auxiliary table. They are commented here for convenience when running
        merged tables.
    """

    url = "https://github.com/FeiyuPan/LakeMLB_datasets/raw/refs/heads/main/datasets/mstraffic.zip"

    def __init__(self, cached_dir: str, csv_name: str = "maryland.csv", mask_name: str = "maryland_mask.pt",
                 force_reload: Optional[bool] = False, transform=None, device=None) -> None:
        self.name = "table_mstraffic"
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
            "maryland.csv",
            "maryland_mask.pt",
            "mask_da.pt",
            "mask_seattle_25pct.pt",
            "mask_seattle_50pct.pt",
            "mask_seattle_75pct.pt",
            "mask_seattle.pt",
            "mstraffic_da.csv",
            "mstraffic_fa.csv",
            "seattle_25pct.csv",
            "seattle_50pct.csv",
            "seattle_75pct.csv",
            "seattle.csv",
        ]

    @property
    def processed_filenames(self):
        base = osp.splitext(self.csv_name)[0]
        return [f"mstraffic_{base}_data.pt"]

    def process(self):
        os.makedirs(self.processed_dir, exist_ok=True)
        csv_path = osp.join(self.raw_dir, self.csv_name)
        masks_path = osp.join(self.raw_dir, self.mask_name)

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
            # "Municipality": ColType.CATEGORICAL,
            "Related Non-Motorist": ColType.CATEGORICAL,
            "At Fault": ColType.CATEGORICAL,
            "Collision Type": ColType.CATEGORICAL,
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
            # aux table cols
            # "OBJECTID": ColType.NUMERICAL,
            # "SE_ANNO_CAD_DATA": ColType.CATEGORICAL,
            # "INCKEY": ColType.NUMERICAL,
            # "COLDETKEY": ColType.NUMERICAL,
            # "REPORTNO": ColType.CATEGORICAL,
            # "STATUS": ColType.CATEGORICAL,
            # "ADDRTYPE": ColType.CATEGORICAL,
            # "INTKEY": ColType.NUMERICAL,
            # # "LOCATION": ColType.CATEGORICAL, #da
            # "EXCEPTRSNCODE": ColType.NUMERICAL,
            # "EXCEPTRSNDESC": ColType.NUMERICAL,
            # "SEVERITYCODE": ColType.NUMERICAL,
            # "SEVERITYDESC": ColType.CATEGORICAL,
            # # "COLLISIONTYPE": ColType.CATEGORICAL, #da
            # "PERSONCOUNT": ColType.NUMERICAL,
            # "PEDCOUNT": ColType.NUMERICAL,
            # "PEDCYLCOUNT": ColType.NUMERICAL,
            # "VEHCOUNT": ColType.NUMERICAL,
            # "INJURIES": ColType.NUMERICAL,
            # "SERIOUSINJURIES": ColType.NUMERICAL,
            # "FATALITIES": ColType.NUMERICAL,
            # "INCDATE": ColType.CATEGORICAL,
            # "INCDTTM": ColType.CATEGORICAL,
            # "JUNCTIONTYPE": ColType.CATEGORICAL,
            # "SDOT_COLCODE": ColType.NUMERICAL,
            # "SDOT_COLDESC": ColType.CATEGORICAL,
            # "INATTENTIONIND": ColType.CATEGORICAL,
            # "UNDERINFL": ColType.CATEGORICAL,
            # # "WEATHER": ColType.CATEGORICAL, #da
            # "ROADCOND": ColType.CATEGORICAL,
            # "LIGHTCOND": ColType.CATEGORICAL,
            # "DIAGRAMLINK": ColType.CATEGORICAL,
            # "REPORTLINK": ColType.CATEGORICAL,
            # "PEDROWNOTGRNT": ColType.NUMERICAL,
            # "SDOTCOLNUM": ColType.NUMERICAL,
            # "SPEEDING": ColType.NUMERICAL,
            # "STCOLCODE": ColType.NUMERICAL,
            # "ST_COLDESC": ColType.CATEGORICAL,
            # "SEGLANEKEY": ColType.NUMERICAL,
            # "CROSSWALKKEY": ColType.NUMERICAL,
            # "HITPARKEDCAR": ColType.CATEGORICAL,
            # "SPDCASENO": ColType.CATEGORICAL,
            # "Source of the collision report": ColType.CATEGORICAL,
            # "Source description": ColType.CATEGORICAL,
            # "Added date": ColType.CATEGORICAL,
            # "Modified date": ColType.CATEGORICAL,
            # "SHAREDMICROMOBILITYCD": ColType.NUMERICAL,
            # "SHAREDMICROMOBILITYDESC": ColType.NUMERICAL,
            # "x": ColType.NUMERICAL,
            # "y": ColType.NUMERICAL,
        }

        df_temp = pd.read_csv(csv_path, nrows=0, low_memory=False)
        dtype_dict = {}
        for col_name, col_type in col_types.items():
            if col_type == ColType.CATEGORICAL and col_name in df_temp.columns:
                dtype_dict[col_name] = str

        df = pd.read_csv(csv_path, low_memory=False, dtype=dtype_dict)

        for col_name, col_type in col_types.items():
            if col_type == ColType.CATEGORICAL and col_name in df.columns:
                df[col_name] = df[col_name].astype(str)

        masks = torch.load(masks_path, weights_only=False)
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
        os.makedirs(self.raw_dir, exist_ok=True)
        path = download_url(self.url, self.raw_dir, "mstraffic.zip")
        extract_zip(path, self.raw_dir)
        os.remove(path)

    def __len__(self):
        return 1

    def __getitem__(self, index: int):
        if index != 0:
            raise IndexError
        return self.data_list[0]
