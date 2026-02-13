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


class NNStocksDataset(Dataset):
    r"""NNStocksDataset is a tabular dataset designed for weakly related (Join-based)
    table scenarios in Data Lake(House) settings, as collected in the `LakeMLB: Data Lake
    Machine Learning Benchmark <https://arxiv.org/abs/2602.10441>`__ paper.

    The dataset focuses on publicly listed companies and comprises two weakly related
    tables: a task table (Nasdaq and NYSE listed companies) and an auxiliary table
    (Wikipedia information). The task table contains information on companies listed on
    Nasdaq and NYSE. The auxiliary table contains company information available on Wikipedia.
    The two tables exhibit a weak association (Join relationship), where information from
    the auxiliary table can be leveraged to enhance machine learning performance on the
    task table. The default task is to predict the industry sector of listed companies
    (a classification task).

    Args:
        cached_dir (str): Root directory where dataset should be saved.
        csv_name (str): Name of the CSV file to use. Default is "nnlist.csv".
        mask_name (str): Name of the mask file. Default is "mask.pt".
        force_reload (bool): If set to `True`, this dataset will be re-process again.
        transform: Optional transform to be applied on the data.
        device: Optional device to move the transformed data to.

    .. parsed-literal::

        Table1: nnlist
        ---------------
            Statics:
            Name        Records     Features
            Size        1,078       11

        Table2: wiki
        ------------------
            Statics:
            Name        Records     Features
            Size        937         22

    Note:
        The columns commented out in col_types (under ``# aux table cols``) belong to
        the auxiliary table. They are commented here for convenience when running
        merged tables.
    """

    url = "https://github.com/FeiyuPan/LakeMLB_datasets/raw/refs/heads/main/datasets/nnstocks.zip"

    def __init__(self, cached_dir: str, csv_name: str = "nnlist.csv", mask_name: str = "mask.pt",
                 force_reload: Optional[bool] = False, transform=None, device=None) -> None:
        self.name = "table_nnstocks"
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
            "mask.pt",
            "nnlist.csv",
            "nnwiki.csv",
            "nnstocks_da.csv",
            "mask_da.pt",
            "t1_enriched_random.csv",
            "t1_enriched_rank1.csv",
            "t1_enriched_rank2.csv",
            "t1_enriched_rank4.csv",
            "t1_enriched_rank8.csv",
            "t1_enriched.csv",
        ]

    @property
    def processed_filenames(self):
        base = osp.splitext(self.csv_name)[0]
        return [f"nnstocks_{base}_data.pt"]

    def process(self):
        os.makedirs(self.processed_dir, exist_ok=True)
        csv_path = osp.join(self.raw_dir, self.csv_name)
        masks_path = osp.join(self.raw_dir, self.mask_name)

        df = pd.read_csv(csv_path)
        col_types = {
            "symbol": ColType.CATEGORICAL,
            "name": ColType.CATEGORICAL,
            "lastsale": ColType.NUMERICAL,
            "netchange": ColType.NUMERICAL,
            "pctchange": ColType.NUMERICAL,
            "volume": ColType.NUMERICAL,
            "marketCap": ColType.NUMERICAL,
            "country": ColType.CATEGORICAL,
            "ipoyear": ColType.NUMERICAL,
            "sector": ColType.CATEGORICAL,
            "url": ColType.CATEGORICAL,
            # aux table cols
            # "wiki_title": ColType.CATEGORICAL,
            # "wiki_url": ColType.CATEGORICAL,
            # "company_type": ColType.CATEGORICAL,
            # "traded_as": ColType.CATEGORICAL,
            # "founded": ColType.CATEGORICAL,
            # "headquarters": ColType.CATEGORICAL,
            # "num_locations": ColType.CATEGORICAL,
            # "area_served": ColType.CATEGORICAL,
            # "key_people": ColType.CATEGORICAL,
            # "services": ColType.CATEGORICAL,
            # "revenue": ColType.CATEGORICAL,
            # "operating_income": ColType.CATEGORICAL,
            # "net_income": ColType.CATEGORICAL,
            # "total_assets": ColType.CATEGORICAL,
            # "total_equity": ColType.CATEGORICAL,
            # "num_employees": ColType.CATEGORICAL,
            # "subsidiaries": ColType.CATEGORICAL,
            # "website": ColType.CATEGORICAL,
            # "founders": ColType.CATEGORICAL,
            # "formerly": ColType.CATEGORICAL,
            # "products": ColType.CATEGORICAL,
            # "isin": ColType.CATEGORICAL,
        }
        masks = torch.load(masks_path, weights_only=False)
        data = TableData(
            df=df,
            col_types=col_types,
            target_col="sector",
            train_mask=masks["train_mask"],
            val_mask=masks["val_mask"],
            test_mask=masks["test_mask"],
        )
        data.save(self.processed_paths[0])

    def download(self):
        os.makedirs(self.raw_dir, exist_ok=True)
        path = download_url(self.url, self.raw_dir, "nnstocks.zip")
        extract_zip(path, self.raw_dir)
        os.remove(path)

    def __len__(self):
        return 1

    def __getitem__(self, index: int):
        if index != 0:
            raise IndexError
        return self.data_list[0]
