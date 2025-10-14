import os
import os.path as osp
from typing import Optional

import pandas as pd

from rllm.types import ColType
from rllm.data.table_data import TableData
from rllm.datasets.dataset import Dataset
from rllm.utils.download import download_url


class Titanic(Dataset):
    r"""The Titanic dataset is a widely-used dataset for machine learning
    and statistical analysis, as featured in the `Titanic: Machine Learning
    from Disaster <https://www.kaggle.com/c/titanic>`__ competition on Kaggle.

    The dataset contains various features related to the passengers
    aboard the Titanic, and the task is to predict whether a
    passenger survived.

    .. PassengerId: Unique identifier for each passenger.
    .. Survived: Survival status (0 = No, 1 = Yes).
    .. Pclass: Passenger class (1 = 1st, 2 = 2nd, 3 = 3rd).
    .. Name: Name of the passenger.
    .. Sex: Gender of the passenger.
    .. Age: Age of the passenger in years.
    .. SibSp: Number of siblings/spouses aboard the Titanic.
    .. Parch: Number of parents/children aboard the Titanic.
    .. Ticket: Ticket number.
    .. Fare: Passenger fare.
    .. Cabin: Cabin number.
    .. Embarked: Port of embarkation
    ..     (C = Cherbourg, Q = Queenstown, S = Southampton).

    Args:
        cached_dir (str): Root directory where dataset should be saved.
        forced_reload (bool): If set to `True`, this dataset will be
            re-process again.

    .. parsed-literal::

        Statics:
        Name   Passengers  Features
        Size   891         12

    """

    url = "https://github.com/datasciencedojo/datasets/raw/master/titanic.csv"

    def __init__(
        self, cached_dir: str, forced_reload: Optional[bool] = False, transform=None
    ) -> None:
        self.name = "titanic"
        root = os.path.join(cached_dir, self.name)
        super().__init__(root, force_reload=forced_reload)
        self.data_list = [TableData.load(self.processed_paths[0])]
        self.transform = transform
        if self.transform is not None:
            self.data_list[0] = self.transform(self.data_list[0])

    @property
    def raw_filenames(self):
        return ["titanic.csv"]

    @property
    def processed_filenames(self):
        return ["data.pt"]

    def process(self):
        r"""
        process data and save to './cached_dir/{dataset}/processed/'.
        """
        os.makedirs(self.processed_dir, exist_ok=True)
        path = osp.join(self.raw_dir, self.raw_filenames[0])
        df = pd.read_csv(path, index_col=["PassengerId"])

        # Note: the order of column in col_types must
        # correspond to the order of column in files,
        # except target column.
        col_types = {  # TODO Use 'Name', 'Ticket' and 'Cabin'.
            "Survived": ColType.CATEGORICAL,
            "Pclass": ColType.CATEGORICAL,
            "Sex": ColType.CATEGORICAL,
            "Age": ColType.NUMERICAL,
            "SibSp": ColType.NUMERICAL,
            "Parch": ColType.NUMERICAL,
            "Fare": ColType.NUMERICAL,
            "Embarked": ColType.CATEGORICAL,
        }
        data = TableData(df=df, col_types=col_types, target_col="Survived")

        data.save(self.processed_paths[0])

    def download(self):
        os.makedirs(self.raw_dir, exist_ok=True)
        download_url(self.url, self.raw_dir, self.raw_filenames[0])

    def __len__(self):
        return 1

    def __getitem__(self, index: int):
        if index != 0:
            raise IndexError
        return self.data_list[index]
