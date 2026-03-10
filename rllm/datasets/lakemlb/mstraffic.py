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

    """

    url = "https://raw.githubusercontent.com/zhengwang100/LakeMLB/main/benckmark/union_based/mstraffic.zip"

    def __init__(self, cached_dir: str, force_reload: Optional[bool] = False, transform=None, device=None
    ) -> None:
        self.name = "table_mstraffic"
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
            "maryland.csv",
            "seattle.csv",
            "mstraffic_da.csv",
            "mstraffic_fa.csv",
            "mask_maryland.pt",
            "mask_da.pt",
        ]

    @property
    def processed_filenames(self):
        return [
            "maryland_data.pt",
            "seattle_data.pt",
            "mstraffic_da_data.pt",
            "mstraffic_fa_data.pt",
        ]

    def process(self):
        os.makedirs(self.processed_dir, exist_ok=True)
        # Maryland Data
        csv_path = osp.join(self.raw_dir, self.raw_filenames[0])
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
            "Municipality": ColType.CATEGORICAL,
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
        }
        maryland_df = pd.read_csv(csv_path, low_memory=False)
        masks_path = osp.join(self.raw_dir, self.raw_filenames[4])
        masks = torch.load(masks_path, weights_only=False)
        TableData(
            df=maryland_df,
            col_types=col_types,
            target_col="Collision Type",
            train_mask=masks["train_mask"],
            val_mask=masks["val_mask"],
            test_mask=masks["test_mask"],
        ).save(self.processed_paths[0])

        # Seattle Data
        csv_path = osp.join(self.raw_dir, self.raw_filenames[1])
        col_types = {
            "OBJECTID": ColType.NUMERICAL,
            "SE_ANNO_CAD_DATA": ColType.CATEGORICAL,
            "INCKEY": ColType.NUMERICAL,
            "COLDETKEY": ColType.NUMERICAL,
            "REPORTNO": ColType.CATEGORICAL,
            "STATUS": ColType.CATEGORICAL,
            "ADDRTYPE": ColType.CATEGORICAL,
            "INTKEY": ColType.NUMERICAL,
            "LOCATION": ColType.CATEGORICAL,
            "EXCEPTRSNCODE": ColType.NUMERICAL,
            "EXCEPTRSNDESC": ColType.NUMERICAL,
            "SEVERITYCODE": ColType.NUMERICAL,
            "SEVERITYDESC": ColType.CATEGORICAL,
            "COLLISIONTYPE": ColType.CATEGORICAL,
            "PERSONCOUNT": ColType.NUMERICAL,
            "PEDCOUNT": ColType.NUMERICAL,
            "PEDCYLCOUNT": ColType.NUMERICAL,
            "VEHCOUNT": ColType.NUMERICAL,
            "INJURIES": ColType.NUMERICAL,
            "SERIOUSINJURIES": ColType.NUMERICAL,
            "FATALITIES": ColType.NUMERICAL,
            "INCDATE": ColType.CATEGORICAL,
            "INCDTTM": ColType.CATEGORICAL,
            "JUNCTIONTYPE": ColType.CATEGORICAL,
            "SDOT_COLCODE": ColType.NUMERICAL,
            "SDOT_COLDESC": ColType.CATEGORICAL,
            "INATTENTIONIND": ColType.CATEGORICAL,
            "UNDERINFL": ColType.CATEGORICAL,
            "WEATHER": ColType.CATEGORICAL,
            "ROADCOND": ColType.CATEGORICAL,
            "LIGHTCOND": ColType.CATEGORICAL,
            "DIAGRAMLINK": ColType.CATEGORICAL,
            "REPORTLINK": ColType.CATEGORICAL,
            "PEDROWNOTGRNT": ColType.NUMERICAL,
            "SDOTCOLNUM": ColType.NUMERICAL,
            "SPEEDING": ColType.NUMERICAL,
            "STCOLCODE": ColType.NUMERICAL,
            "ST_COLDESC": ColType.CATEGORICAL,
            "SEGLANEKEY": ColType.NUMERICAL,
            "CROSSWALKKEY": ColType.NUMERICAL,
            "HITPARKEDCAR": ColType.CATEGORICAL,
            "SPDCASENO": ColType.CATEGORICAL,
            "Source of the collision report": ColType.CATEGORICAL,
            "Source description": ColType.CATEGORICAL,
            "Added date": ColType.CATEGORICAL,
            "Modified date": ColType.CATEGORICAL,
            "SHAREDMICROMOBILITYCD": ColType.NUMERICAL,
            "SHAREDMICROMOBILITYDESC": ColType.NUMERICAL,
            "x": ColType.NUMERICAL,
            "y": ColType.NUMERICAL,
        }
        seattle_df = pd.read_csv(csv_path, low_memory=False)
        TableData(
            df=seattle_df,
            col_types=col_types,
            target_col="COLLISIONTYPE",
        ).save(self.processed_paths[1])

        # Merged Data(DA)
        csv_path = osp.join(self.raw_dir, self.raw_filenames[2])
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
            "Municipality": ColType.CATEGORICAL,
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
            "OBJECTID": ColType.NUMERICAL,
            "SE_ANNO_CAD_DATA": ColType.CATEGORICAL,
            "INCKEY": ColType.NUMERICAL,
            "COLDETKEY": ColType.NUMERICAL,
            "REPORTNO": ColType.CATEGORICAL,
            "STATUS": ColType.CATEGORICAL,
            "ADDRTYPE": ColType.CATEGORICAL,
            "INTKEY": ColType.NUMERICAL,
            # "LOCATION": ColType.CATEGORICAL, #da
            "EXCEPTRSNCODE": ColType.NUMERICAL,
            "EXCEPTRSNDESC": ColType.NUMERICAL,
            "SEVERITYCODE": ColType.NUMERICAL,
            "SEVERITYDESC": ColType.CATEGORICAL,
            # "COLLISIONTYPE": ColType.CATEGORICAL, #da
            "PERSONCOUNT": ColType.NUMERICAL,
            "PEDCOUNT": ColType.NUMERICAL,
            "PEDCYLCOUNT": ColType.NUMERICAL,
            "VEHCOUNT": ColType.NUMERICAL,
            "INJURIES": ColType.NUMERICAL,
            "SERIOUSINJURIES": ColType.NUMERICAL,
            "FATALITIES": ColType.NUMERICAL,
            "INCDATE": ColType.CATEGORICAL,
            "INCDTTM": ColType.CATEGORICAL,
            "JUNCTIONTYPE": ColType.CATEGORICAL,
            "SDOT_COLCODE": ColType.NUMERICAL,
            "SDOT_COLDESC": ColType.CATEGORICAL,
            "INATTENTIONIND": ColType.CATEGORICAL,
            "UNDERINFL": ColType.CATEGORICAL,
            # "WEATHER": ColType.CATEGORICAL, #da
            "ROADCOND": ColType.CATEGORICAL,
            "LIGHTCOND": ColType.CATEGORICAL,
            "DIAGRAMLINK": ColType.CATEGORICAL,
            "REPORTLINK": ColType.CATEGORICAL,
            "PEDROWNOTGRNT": ColType.NUMERICAL,
            "SDOTCOLNUM": ColType.NUMERICAL,
            "SPEEDING": ColType.NUMERICAL,
            "STCOLCODE": ColType.NUMERICAL,
            "ST_COLDESC": ColType.CATEGORICAL,
            "SEGLANEKEY": ColType.NUMERICAL,
            "CROSSWALKKEY": ColType.NUMERICAL,
            "HITPARKEDCAR": ColType.CATEGORICAL,
            "SPDCASENO": ColType.CATEGORICAL,
            "Source of the collision report": ColType.CATEGORICAL,
            "Source description": ColType.CATEGORICAL,
            "Added date": ColType.CATEGORICAL,
            "Modified date": ColType.CATEGORICAL,
            "SHAREDMICROMOBILITYCD": ColType.NUMERICAL,
            "SHAREDMICROMOBILITYDESC": ColType.NUMERICAL,
            "x": ColType.NUMERICAL,
            "y": ColType.NUMERICAL,
        }
        mstraffic_da_df = pd.read_csv(csv_path, low_memory=False)
        masks_path = osp.join(self.raw_dir, self.raw_filenames[5])
        masks = torch.load(masks_path, weights_only=False)
        TableData(
            df=mstraffic_da_df,
            col_types=col_types,
            target_col="Collision Type",
            train_mask=masks["train_mask"],
            val_mask=masks["val_mask"],
            test_mask=masks["test_mask"],
        ).save(self.processed_paths[2])

        # Merged Data(FA)
        csv_path = osp.join(self.raw_dir, self.raw_filenames[3])
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
            "Municipality": ColType.CATEGORICAL,
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
            "OBJECTID": ColType.NUMERICAL,
            "SE_ANNO_CAD_DATA": ColType.CATEGORICAL,
            "INCKEY": ColType.NUMERICAL,
            "COLDETKEY": ColType.NUMERICAL,
            "REPORTNO": ColType.CATEGORICAL,
            "STATUS": ColType.CATEGORICAL,
            "ADDRTYPE": ColType.CATEGORICAL,
            "INTKEY": ColType.NUMERICAL,
            "LOCATION": ColType.CATEGORICAL,
            "EXCEPTRSNCODE": ColType.NUMERICAL,
            "EXCEPTRSNDESC": ColType.NUMERICAL,
            "SEVERITYCODE": ColType.NUMERICAL,
            "SEVERITYDESC": ColType.CATEGORICAL,
            "COLLISIONTYPE": ColType.CATEGORICAL,
            "PERSONCOUNT": ColType.NUMERICAL,
            "PEDCOUNT": ColType.NUMERICAL,
            "PEDCYLCOUNT": ColType.NUMERICAL,
            "VEHCOUNT": ColType.NUMERICAL,
            "INJURIES": ColType.NUMERICAL,
            "SERIOUSINJURIES": ColType.NUMERICAL,
            "FATALITIES": ColType.NUMERICAL,
            "INCDATE": ColType.CATEGORICAL,
            "INCDTTM": ColType.CATEGORICAL,
            "JUNCTIONTYPE": ColType.CATEGORICAL,
            "SDOT_COLCODE": ColType.NUMERICAL,
            "SDOT_COLDESC": ColType.CATEGORICAL,
            "INATTENTIONIND": ColType.CATEGORICAL,
            "UNDERINFL": ColType.CATEGORICAL,
            "WEATHER": ColType.CATEGORICAL,
            "ROADCOND": ColType.CATEGORICAL,
            "LIGHTCOND": ColType.CATEGORICAL,
            "DIAGRAMLINK": ColType.CATEGORICAL,
            "REPORTLINK": ColType.CATEGORICAL,
            "PEDROWNOTGRNT": ColType.NUMERICAL,
            "SDOTCOLNUM": ColType.NUMERICAL,
            "SPEEDING": ColType.NUMERICAL,
            "STCOLCODE": ColType.NUMERICAL,
            "ST_COLDESC": ColType.CATEGORICAL,
            "SEGLANEKEY": ColType.NUMERICAL,
            "CROSSWALKKEY": ColType.NUMERICAL,
            "HITPARKEDCAR": ColType.CATEGORICAL,
            "SPDCASENO": ColType.CATEGORICAL,
            "Source of the collision report": ColType.CATEGORICAL,
            "Source description": ColType.CATEGORICAL,
            "Added date": ColType.CATEGORICAL,
            "Modified date": ColType.CATEGORICAL,
            "SHAREDMICROMOBILITYCD": ColType.NUMERICAL,
            "SHAREDMICROMOBILITYDESC": ColType.NUMERICAL,
            "x": ColType.NUMERICAL,
            "y": ColType.NUMERICAL,
        }
        mstraffic_fa_df = pd.read_csv(csv_path, low_memory=False)
        masks_path = osp.join(self.raw_dir, self.raw_filenames[4])
        masks = torch.load(masks_path, weights_only=False)
        TableData(
            df=mstraffic_fa_df,
            col_types=col_types,
            target_col="Collision Type",
            train_mask=masks["train_mask"],
            val_mask=masks["val_mask"],
            test_mask=masks["test_mask"],
        ).save(self.processed_paths[3])

    def download(self):
        os.makedirs(self.raw_dir, exist_ok=True)
        path = download_url(self.url, self.raw_dir, "mstraffic.zip")
        extract_zip(path, self.raw_dir)
        os.remove(path)

    def __len__(self):
        return 4

    def __getitem__(self, index: int):
        if index < 0 or index >= len(self.data_list):
            raise IndexError
        return self.data_list[index]
