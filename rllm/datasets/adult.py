import os
import os.path as osp
from typing import Optional

import pandas as pd

from rllm.types import ColType
from rllm.data.table_data import TableData
from rllm.datasets.dataset import Dataset
from rllm.utils.download import download_url


class Adult(Dataset):
    r"""The Adult dataset is a dataset from a classic data mining project,
    which was extracted from the `1994 Census database
    <https://archive.ics.uci.edu/dataset/2/adult>`__.

    The dataset encompasses a variety of features pertaining to adults and
    their income. The primary objective is to predict whether an individual's
    annual income surpasses $50,000.

    .. Age: Age of the individual.
    .. Workclass: Type of industry (Private, Self-emp-not-inc, Self-emp-inc,
    ..     Federal-gov, Local-gov, State-gov, Without-pay, Never-worked).
    .. fnlwgt: The number of people the census believes have this job.
    .. Education: The highest level of education achieved (Bachelors,
    ..     Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th,
    ..     7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool).
    .. Education-Num: A numeric version of Education.
    .. Marital-Status: Marital status of the individual (Married-civ-spouse,
    ..     Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse).
    .. Occupation: The kind of work individuals perform (Tech-support,
    ..     Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty,
    ..     Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing,
    ..     Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces).
    .. Relationship: Relationship to head-of-household (Wife, Own-child, Husband,
    ..     Not-in-family, Other-relative, Unmarried).
    .. Race: Race of the individual (White, Asian-Pac-Islander,
    ..     Amer-Indian-Eskimo, Other, Black).
    .. Sex: Gender of the individual.
    .. Capital-Gain: Total capital gains.
    .. Capital-Loss: Total capital losses.
    .. Hours-per-week: Average hours worked per week.
    .. Native-Country: Country of origin of the individual.
    .. Target: Income level.

    Args:
        cached_dir (str): Root directory where dataset should be saved.
        forced_reload (bool): If set to `True`, this dataset will be
            re-process again.

    .. parsed-literal::

        Statics:
        Name   Individuals  Features
        Size   48842        14

    """

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

    def __init__(self, cached_dir: str, forced_reload: Optional[bool] = False) -> None:
        self.name = "adult"
        root = os.path.join(cached_dir, self.name)
        super().__init__(root, force_reload=forced_reload)
        self.data_list = [TableData.load(self.processed_paths[0])]

    @property
    def raw_filenames(self):
        return ["adult.csv"]

    @property
    def processed_filenames(self):
        return ["data.pt"]

    def process(self):
        r"""
        process data and save to './cached_dir/{dataset}/processed/'.
        """
        os.makedirs(self.processed_dir, exist_ok=True)
        path = osp.join(self.raw_dir, self.raw_filenames[0])

        # Note: the order of column in col_types must
        # correspond to the order of column in files,
        # except target column.
        col_types = {
            "age": ColType.NUMERICAL,
            "workclass": ColType.CATEGORICAL,
            "fnlwgt": ColType.NUMERICAL,
            "education": ColType.CATEGORICAL,
            "educational-num": ColType.NUMERICAL,
            "marital-status": ColType.CATEGORICAL,
            "occupation": ColType.CATEGORICAL,
            "relationship": ColType.CATEGORICAL,
            "race": ColType.CATEGORICAL,
            "gender": ColType.CATEGORICAL,
            "capital-gain": ColType.NUMERICAL,
            "capital-loss": ColType.NUMERICAL,
            "hours-per-week": ColType.NUMERICAL,
            "native-country": ColType.CATEGORICAL,
            "income": ColType.CATEGORICAL,
        }

        df = pd.read_csv(path, header=None, names=list(col_types.keys()))
        data = TableData(
            df=df,
            col_types=col_types,
            target_col="income",
        )

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
