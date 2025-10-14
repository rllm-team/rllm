import os
import os.path as osp
from typing import Optional

import pandas as pd

from rllm.types import ColType
from rllm.data.table_data import TableData
from rllm.datasets.dataset import Dataset
from rllm.utils.download import download_url


class ChurnModelling(Dataset):
    r"""`The Churn Modelling dataset <https://www.kaggle.com/shrutimechlearn/
    churn-modelling>`__ is used to predict which customers are
    likely to churn from the organization by analyzing various attributes and
    applying machine learning and deep learning techniques.

    Customer churn refers to when a customer (player, subscriber, user, etc.)
    ceases their relationship with a company. Online businesses typically treat
    a customer as churned once a particular amount of time has elapsed since
    the customer's last interaction with the site or service.

    Customer churn occurs when customers or subscribers stop doing business
    with a company or service, also known as customer attrition. It is also
    referred to as loss of clients or customers. Similar to predicting
    employee turnover, we are going to predict customer churn using this
    dataset.

    The dataset encompasses a variety of features pertaining to customers and
    their interactions with the company. The primary objective is to predict
    whether a customer will churn.

    .. RowNumber: Row number.
    .. CustomerId: Unique identifier for the customer.
    .. Surname: Surname of the customer.
    .. CreditScore: Credit score of the customer.
    .. Geography: Country of the customer (France, Spain, Germany).
    .. Gender: Gender of the customer (Male, Female).
    .. Age: Age of the customer.
    .. Tenure: Number of years the customer has been with the company.
    .. Balance: Account balance of the customer.
    .. NumOfProducts: Number of products the customer has with the company.
    .. HasCrCard: Does the customer have a credit card? (0 = No, 1 = Yes).
    .. IsActiveMember: Is the customer an active member? (0 = No, 1 = Yes).
    .. EstimatedSalary: Estimated salary of the customer.
    .. Exited: Did the customer churn? (0 = No, 1 = Yes).

    Args:
        cached_dir (str): Root directory where dataset should be saved.
        forced_reload (bool): If set to `True`, this dataset will be
            re-processed again.

    .. parsed-literal::

        Statics:
        Name   Customers   Features
        Size   10000       14

    """

    url = "https://raw.githubusercontent.com/sharmaroshan/Churn-Modelling-Dataset/master/Churn_Modelling.csv"

    def __init__(self, cached_dir: str, forced_reload: Optional[bool] = False) -> None:
        self.name = "churn"
        root = os.path.join(cached_dir, self.name)
        super().__init__(root, force_reload=forced_reload)
        self.data_list = [TableData.load(self.processed_paths[0])]

    @property
    def raw_filenames(self):
        return ["churn.csv"]

    @property
    def processed_filenames(self):
        return ["data.pt"]

    def process(self):
        r"""
        process data and save to './cached_dir/{dataset}/processed/'.
        """
        os.makedirs(self.processed_dir, exist_ok=True)
        path = osp.join(self.raw_dir, self.raw_filenames[0])
        df = pd.read_csv(path, index_col=["RowNumber"])

        # Note: the order of column in col_types must
        # correspond to the order of column in files,
        # except target column.
        col_types = {
            "CreditScore": ColType.NUMERICAL,
            "Geography": ColType.CATEGORICAL,
            "Gender": ColType.CATEGORICAL,
            "Age": ColType.NUMERICAL,
            "Tenure": ColType.NUMERICAL,
            "Balance": ColType.NUMERICAL,
            "NumOfProducts": ColType.NUMERICAL,
            "HasCrCard": ColType.NUMERICAL,
            "IsActiveMember": ColType.CATEGORICAL,
            "EstimatedSalary": ColType.NUMERICAL,
            "Exited": ColType.CATEGORICAL,
        }
        data = TableData(
            df=df,
            col_types=col_types,
            target_col="Exited",
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
