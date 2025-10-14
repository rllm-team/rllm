import os
import os.path as osp
from typing import Optional

import pandas as pd

from rllm.types import ColType
from rllm.data.table_data import TableData
from rllm.datasets.dataset import Dataset
from rllm.utils.download import download_url
from rllm.utils.extract import extract_zip


class BankMarketing(Dataset):
    r"""`The Bank Marketing dataset <https://archive.ics.uci.edu/dataset/
    222/bank+marketing>`__ is related to direct marketing campaigns of
    a Portuguese banking institution. The marketing campaigns were based on
    phone calls. Often, more than one contact to the same client was required
    in order to assess if the product (bank term deposit) would be subscribed.
    The classification goal is to predict if the client will subscribe to a
    term deposit.

    The dataset encompasses a variety of features pertaining to clients and
    their banking information. The primary objective is to predict whether
    a client will subscribe to a term deposit.

    .. Age: Age of the client.
    .. Job: Type of job (admin, blue-collar, entrepreneur, housemaid,
    ..     management, retired, self-employed, services, student, technician,
    ..     unemployed, unknown).
    .. Marital: Marital status of the client (divorced, married, single).
    .. Education: The highest level of education achieved (unknown, secondary,
    ..     primary, tertiary).
    .. Default: Has credit in default?
    .. Balance: Average yearly balance, in euros.
    .. Housing: Has housing loan?
    .. Loan: Has personal loan?
    .. Contact: Contact communication type (unknown, telephone, cellular).
    .. Day: Last contact day of the month.
    .. Month: Last contact month of the year (jan, feb, mar, apr, may, jun, jul,
    ..     aug, sep, oct, nov, dec).
    .. Duration: Last contact duration, in seconds.
    .. Campaign: Number of contacts performed during this campaign and for this
    ..     client.
    .. Pdays: Number of days that passed by after the client was last contacted
    ..     from a previous campaign.
    .. Previous: Number of contacts performed before this campaign and for this
    ..     client.
    .. Poutcome: Outcome of the previous marketing campaign (unknown, other,
    ..     failure, success).
    .. Target: Has the client subscribed a term deposit?

    Args:
        cached_dir (str): Root directory where dataset should be saved.
        forced_reload (bool): If set to `True`, this dataset will be
            re-processed again.

    .. parsed-literal::

        Statics:
        Name   Clients  Features
        Size   45211    16

    """

    url = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"

    def __init__(self, cached_dir: str, forced_reload: Optional[bool] = False) -> None:
        self.name = "bank_marketing"
        root = os.path.join(cached_dir, self.name)
        super().__init__(root, force_reload=forced_reload)
        self.data_list = [TableData.load(self.processed_paths[0])]

    @property
    def raw_filenames(self):
        return ["bank-full.csv"]

    @property
    def processed_filenames(self):
        return ["data.pt"]

    def process(self):
        r"""
        process data and save to './cached_dir/{dataset}/processed/'.
        """
        os.makedirs(self.processed_dir, exist_ok=True)
        path = osp.join(self.raw_dir, self.raw_filenames[0])
        df = pd.read_csv(path, sep=";", quotechar='"')

        # Note: the order of column in col_types must
        # correspond to the order of column in files,
        # except target column.
        col_types = {
            "age": ColType.NUMERICAL,
            "job": ColType.CATEGORICAL,
            "marital": ColType.CATEGORICAL,
            "education": ColType.CATEGORICAL,
            "default": ColType.CATEGORICAL,
            "balance": ColType.NUMERICAL,
            "housing": ColType.CATEGORICAL,
            "loan": ColType.CATEGORICAL,
            "contact": ColType.CATEGORICAL,
            "day": ColType.NUMERICAL,
            "month": ColType.CATEGORICAL,
            "duration": ColType.NUMERICAL,
            "campaign": ColType.NUMERICAL,
            "pdays": ColType.NUMERICAL,
            "previous": ColType.NUMERICAL,
            "poutcome": ColType.CATEGORICAL,
            "y": ColType.CATEGORICAL,
        }
        data = TableData(
            df=df,
            col_types=col_types,
            target_col="y",
        )

        data.save(self.processed_paths[0])

    def download(self):
        os.makedirs(self.raw_dir, exist_ok=True)
        download_url(self.url, self.raw_dir, "bank+marketing.zip")
        extract_zip(osp.join(self.raw_dir, "bank+marketing.zip"), self.raw_dir)
        extract_zip(osp.join(self.raw_dir, "bank.zip"), self.raw_dir)

    def __len__(self):
        return 1

    def __getitem__(self, index: int):
        if index != 0:
            raise IndexError
        return self.data_list[index]
