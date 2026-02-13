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


class LHStocksDataset(Dataset):
    r"""LHStocksDataset is a tabular dataset designed for weakly related (Join-based)
    table scenarios in Data Lake(House) settings, as collected in the `LakeMLB: Data Lake
    Machine Learning Benchmark <https://arxiv.org/abs/2602.10441>`__ paper.

    The dataset focuses on publicly listed companies and comprises two weakly related
    tables: a task table (London and Hong Kong listed companies) and an auxiliary table
    (Wikipedia information). The task table contains information on companies listed
    in London and Hong Kong. The auxiliary table contains company information available
    on Wikipedia. The two tables exhibit a weak association (Join relationship), where
    information from the auxiliary table can be leveraged to enhance machine learning
    performance on the task table. The default task is to predict the industry sector
    of listed companies (a classification task).

    Args:
        cached_dir (str): Root directory where dataset should be saved.
        csv_name (str): Name of the CSV file to use. Default is "lhlist.csv".
        mask_name (str): Name of the mask file. Default is "stocks_lh_mask.pt".
        force_reload (bool): If set to `True`, this dataset will be re-process again.
        transform: Optional transform to be applied on the data.
        device: Optional device to move the transformed data to.

    .. parsed-literal::

        Table1: lhlist
        ---------------
            Statics:
            Name        Records     Features
            Size        1,078       16

        Table2: wiki
        ------------------
            Statics:
            Name        Records     Features
            Size        937         21

    Note:
        The columns commented out in col_types (under ``# aux table cols``) belong to
        the auxiliary table. They are commented here for convenience when running
        merged tables.
    """

    url = "https://github.com/FeiyuPan/LakeMLB_datasets/raw/refs/heads/main/datasets/lhstocks.zip"

    def __init__(self, cached_dir: str, csv_name: str = "lhlist.csv", mask_name: str = "stocks_lh_mask.pt",
                 force_reload: Optional[bool] = False, transform=None, device=None) -> None:
        self.name = "table_lhstocks"
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
            "lhlist.csv",
            "lhwiki.csv",
            "stocks_lh_mask.pt",
            "wiki_lh_mask.pt",
            "stocks_lh_enriched.csv",
            "lhstocks_da.csv",
            "mask_da.pt",
            "merged_lh.csv",
        ]

    @property
    def processed_filenames(self):
        base = osp.splitext(self.csv_name)[0]
        return [f"lhstocks_{base}_data.pt"]

    def process(self):
        os.makedirs(self.processed_dir, exist_ok=True)
        csv_path = osp.join(self.raw_dir, self.csv_name)
        masks_path = osp.join(self.raw_dir, self.mask_name)

        df = pd.read_csv(csv_path)
        col_types = {
            "Source": ColType.CATEGORICAL,
            "Company Name": ColType.CATEGORICAL,
            "ICB Industry": ColType.CATEGORICAL,
            "Admission Date": ColType.CATEGORICAL,
            "Country of Incorporation": ColType.CATEGORICAL,
            "World Region": ColType.CATEGORICAL,
            "Market": ColType.CATEGORICAL,
            "International Issuer": ColType.CATEGORICAL,
            "Company Market Cap (￡m)": ColType.NUMERICAL,
            "Stock Code": ColType.CATEGORICAL,
            "Listing Status": ColType.CATEGORICAL,
            "Director's English Name": ColType.CATEGORICAL,
            "Capacity": ColType.CATEGORICAL,
            "Position": ColType.CATEGORICAL,
            "Appointment Date (yyyy-mm-dd)": ColType.CATEGORICAL,
            "Resignation Date (yyyy-mm-dd)": ColType.CATEGORICAL,
            # aux table cols
            # "wiki_title": ColType.CATEGORICAL,
            # "wiki_url": ColType.CATEGORICAL,
            # "area_served": ColType.CATEGORICAL,
            # "company_type": ColType.CATEGORICAL,
            # "founded": ColType.CATEGORICAL,
            # "founders": ColType.CATEGORICAL,
            # "headquarters": ColType.CATEGORICAL,
            # "industry": ColType.CATEGORICAL,
            # "key_people": ColType.CATEGORICAL,
            # "net_income": ColType.CATEGORICAL,
            # "num_employees": ColType.CATEGORICAL,
            # "operating_income": ColType.CATEGORICAL,
            # "owner": ColType.CATEGORICAL,
            # "parent": ColType.CATEGORICAL,
            # "products": ColType.CATEGORICAL,
            # "revenue": ColType.CATEGORICAL,
            # "subsidiaries": ColType.CATEGORICAL,
            # "total_assets": ColType.CATEGORICAL,
            # "total_equity": ColType.CATEGORICAL,
            # "traded_as": ColType.CATEGORICAL,
            # "website": ColType.CATEGORICAL,
        }
        masks = torch.load(masks_path, weights_only=False)
        data = TableData(
            df=df,
            col_types=col_types,
            target_col="ICB Industry",
            train_mask=masks["train_mask"],
            val_mask=masks["val_mask"],
            test_mask=masks["test_mask"],
        )
        data.save(self.processed_paths[0])

    def download(self):
        os.makedirs(self.raw_dir, exist_ok=True)
        path = download_url(self.url, self.raw_dir, "lhstocks.zip")
        extract_zip(path, self.raw_dir)
        os.remove(path)

    def __len__(self):
        return 1

    def __getitem__(self, index: int):
        if index != 0:
            raise IndexError
        return self.data_list[0]
