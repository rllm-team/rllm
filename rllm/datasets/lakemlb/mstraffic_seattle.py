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

class MSTrafficSeattleDataset(Dataset):
    r"""The MS-Traffic dataset for collision type prediction.
    
    This dataset contains traffic crash records for Seattle. The task is to
    predict the 'Collision Type'.

    The raw data is downloaded as a zip file and contains multiple files.
    This class processes 'Seattle.csv' and uses masks from 'T1_mask.pt'.
    """
    
    #url = "https://github.com/FeiyuPan/LakeMLB_datasets/raw/refs/heads/main/datasets/Crash20K.zip"
    
    archive_filename = "Crash20K.zip"

    def __init__(self, cached_dir: str, force_reload: Optional[bool] = False, transform=None, device=None) -> None:
        self.name = "mstraffic-seattle"
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
        return ["seattle.csv", "T2_mask.pt"]

    @property
    def processed_filenames(self) -> List[str]:
        return ["mstraffic_seattle_data.pt"]

    def process(self):
        """Processes the raw csv and mask files into a single TableData object."""
        os.makedirs(self.processed_dir, exist_ok=True)
        
        csv_path = osp.join(self.raw_dir, self.raw_filenames[0])
        masks_path = osp.join(self.raw_dir, self.raw_filenames[1])

        df = pd.read_csv(csv_path)
        col_types = {
            #"SE_ANNO_CAD_DATA": ColType.CATEGORICAL,
            "REPORTNO": ColType.CATEGORICAL,
            #"STATUS": ColType.CATEGORICAL,
            "ADDRTYPE": ColType.CATEGORICAL,
            "LOCATION": ColType.CATEGORICAL,
            "SEVERITYCODE": ColType.CATEGORICAL,
            "SEVERITYDESC": ColType.CATEGORICAL,
            "INCDATE": ColType.CATEGORICAL,
            "INCDTTM": ColType.CATEGORICAL,
            "JUNCTIONTYPE": ColType.CATEGORICAL,
            "UNDERINFL": ColType.CATEGORICAL,
            "WEATHER": ColType.CATEGORICAL,
            "ROADCOND": ColType.CATEGORICAL,
            "LIGHTCOND": ColType.CATEGORICAL,
            "DIAGRAMLINK": ColType.CATEGORICAL,
            "REPORTLINK": ColType.CATEGORICAL,
            #"PEDROWNOTGRNT": ColType.CATEGORICAL,
            "SPEEDING": ColType.CATEGORICAL,
            "HITPARKEDCAR": ColType.CATEGORICAL,
            "SPDCASENO": ColType.CATEGORICAL,
            "Source of the collision report": ColType.CATEGORICAL,
            "Source description": ColType.CATEGORICAL,
            "Added date": ColType.CATEGORICAL,
            "Modified date": ColType.CATEGORICAL,
            #"SHAREDMICROMOBILITYCD": ColType.CATEGORICAL,
            #"SHAREDMICROMOBILITYDESC": ColType.CATEGORICAL,
            "COLLISIONTYPE": ColType.CATEGORICAL,
            "OBJECTID": ColType.NUMERICAL,
            "INCKEY": ColType.NUMERICAL,
            "COLDETKEY": ColType.NUMERICAL,
            "INTKEY": ColType.NUMERICAL,
            "PERSONCOUNT": ColType.NUMERICAL,
            "PEDCOUNT": ColType.NUMERICAL,
            "PEDCYLCOUNT": ColType.NUMERICAL,
            "VEHCOUNT": ColType.NUMERICAL,
            "INJURIES": ColType.NUMERICAL,
            "SERIOUSINJURIES": ColType.NUMERICAL,
            "FATALITIES": ColType.NUMERICAL,
            "CROSSWALKKEY": ColType.NUMERICAL,
            "x": ColType.NUMERICAL,
            "y": ColType.NUMERICAL,
        }
        
        masks = torch.load(masks_path) # weights_only=False 是默认值，可以省略
        
        data = TableData(
            df=df,
            col_types=col_types,
            target_col="COLLISIONTYPE",
            train_mask=masks["train_mask"],
            val_mask=masks["val_mask"],
            test_mask=masks["test_mask"],
        )
        
        # 保存路径 self.processed_paths[0] 的用法完全正确
        data.save(self.processed_paths[0])

    def download(self):
        """Downloads the zip archive, extracts it, and removes the archive."""
        os.makedirs(self.raw_dir, exist_ok=True)
        # 3. (已修复) 使用类属性来引用文件名
        path = download_url(self.url, self.raw_dir, self.archive_filename)
        extract_zip(path, self.raw_dir)
        os.remove(path)

    def __len__(self) -> int:
        return len(self.data_list) # 返回 1 或 len(self.data_list) 都可以

    def __getitem__(self, index: int) -> TableData:
        if index >= len(self.data_list):
            raise IndexError
        return self.data_list[index]