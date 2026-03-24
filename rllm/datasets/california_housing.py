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
    r"""The California Housing dataset is a classic regression dataset,
    fetched from ``sklearn.datasets.fetch_california_housing``.

    The dataset contains census block-group level housing information from
    California. The objective is to predict the median house value.

    .. MedInc: Median income in a block group.
    .. HouseAge: Median house age in a block group.
    .. AveRooms: Average number of rooms per household.
    .. AveBedrms: Average number of bedrooms per household.
    .. Population: Block-group population.
    .. AveOccup: Average household occupancy.
    .. Latitude: Latitude of the block group.
    .. Longitude: Longitude of the block group.
    .. Target: Median house value (MedHouseVal).

    Args:
        cached_dir (str): Root directory where dataset should be saved.
        forced_reload (bool): If set to `True`, this dataset will be
            re-process again.

    .. parsed-literal::

        Statics:
        Name                 Block Groups  Features
        Size                 20640         8

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
