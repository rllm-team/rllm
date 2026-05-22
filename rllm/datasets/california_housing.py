import os
import os.path as osp
from typing import Optional

import pandas as pd

from rllm.types import ColType
from rllm.data.table_data import TableData
from rllm.datasets.dataset import Dataset

try:
    from sklearn.datasets import fetch_california_housing
except Exception as e:
    raise ImportError(
        "scikit-learn is required for CaliforniaHousing dataset. Please install scikit-learn."
    ) from e


class CaliforniaHousing(Dataset):
    r"""California Housing (Regression) dataset via scikit-learn.

    Fetches the California Housing dataset using
    ``sklearn.datasets.fetch_california_housing`` and converts it into a
    ``TableData`` object with appropriate column types.

    Features (8 numerical):
    - MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude

    Target (numerical):
    - MedHouseVal

    Args:
            cached_dir (str): Root directory where dataset should be saved.
            forced_reload (bool): If set to True, re-process and overwrite.
    """

    def __init__(self, cached_dir: str, forced_reload: Optional[bool] = False) -> None:
        self.name = "california_housing"
        root = os.path.join(cached_dir, self.name)
        super().__init__(root, force_reload=forced_reload)
        self.data_list = [TableData.load(self.processed_paths[0])]

    @property
    def raw_filenames(self):
        return ["california_housing.csv"]

    @property
    def processed_filenames(self):
        return ["data.pt"]

    def download(self):
        os.makedirs(self.raw_dir, exist_ok=True)

        bunch = fetch_california_housing()
        df = pd.DataFrame(data=bunch.data, columns=bunch.feature_names)
        df["MedHouseVal"] = bunch.target

        df.to_csv(osp.join(self.raw_dir, self.raw_filenames[0]), index=False)

    def process(self):
        os.makedirs(self.processed_dir, exist_ok=True)
        path = osp.join(self.raw_dir, self.raw_filenames[0])
        df = pd.read_csv(path)

        col_types = {
            "MedInc": ColType.NUMERICAL,
            "HouseAge": ColType.NUMERICAL,
            "AveRooms": ColType.NUMERICAL,
            "AveBedrms": ColType.NUMERICAL,
            "Population": ColType.NUMERICAL,
            "AveOccup": ColType.NUMERICAL,
            "Latitude": ColType.NUMERICAL,
            "Longitude": ColType.NUMERICAL,
            "MedHouseVal": ColType.NUMERICAL,
        }

        data = TableData(
            df=df,
            col_types=col_types,
            target_col="MedHouseVal",
        )

        data.save(self.processed_paths[0])

    def __len__(self):
        return 1

    def __getitem__(self, index: int):
        if index != 0:
            raise IndexError
        return self.data_list[index]
