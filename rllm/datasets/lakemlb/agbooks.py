from typing import List, Optional
import os

import pandas as pd
import torch

from rllm.types import ColType
from rllm.data.table_data import TableData
from rllm.datasets.dataset import Dataset
from rllm.utils.download import download_url
from rllm.utils.extract import extract_zip


class AGBooksDataset(Dataset):
    r"""A LakeMLB join-based dataset for book-category classification.

    The task table contains Amazon book metadata and the auxiliary table
    contains Goodreads metadata. The processed dataset also provides the
    corresponding data-augmentation (DA) and feature-augmentation (FA)
    tables.

    Args:
        cached_dir (str): Root directory where the dataset is stored.
        force_reload (bool): Whether to process the raw files again.
        transform: Optional transform applied to each processed table.
        device: Optional device for transformed tables.
    """

    url = (
        "https://media.githubusercontent.com/media/zhengwang100/LakeMLB/"
        "main/benckmark/join_based/agbooks.zip"
    )

    def __init__(
        self,
        cached_dir: str,
        force_reload: Optional[bool] = False,
        transform=None,
        device=None,
    ) -> None:
        self.name = "table_agbooks"
        root = os.path.join(cached_dir, self.name)
        super().__init__(root, force_reload=force_reload)

        self.data_list: List[TableData] = [
            TableData.load(path) for path in self.processed_paths
        ]
        self.transform = transform
        if self.transform is not None:
            for i, data in enumerate(self.data_list):
                self.data_list[i] = (
                    self.transform(data).to(device)
                    if device is not None
                    else self.transform(data)
                )

    @property
    def raw_filenames(self):
        return [
            "amazon.csv",
            "goodreads.csv",
            "agbooks_da.csv",
            "amazon_fa.csv",
            "mask_amazon.pt",
            "mask_da.pt",
        ]

    @property
    def processed_filenames(self):
        return [
            "amazon_data.pt",
            "goodreads_data.pt",
            "agbooks_da_data.pt",
            "agbooks_fa_data.pt",
        ]

    def process(self):
        os.makedirs(self.processed_dir, exist_ok=True)

        amazon_col_types = {
            "parent_asin": ColType.CATEGORICAL,
            "title": ColType.CATEGORICAL,
            "main_category": ColType.CATEGORICAL,
            "average_rating": ColType.NUMERICAL,
            "rating_number": ColType.NUMERICAL,
            "price": ColType.CATEGORICAL,
            "store": ColType.CATEGORICAL,
            "features": ColType.CATEGORICAL,
            "description": ColType.CATEGORICAL,
            "details": ColType.CATEGORICAL,
            "categories": ColType.CATEGORICAL,
        }
        goodreads_col_types = {
            "book_id": ColType.CATEGORICAL,
            "title": ColType.CATEGORICAL,
            "title_without_series": ColType.CATEGORICAL,
            "average_rating": ColType.NUMERICAL,
            "ratings_count": ColType.NUMERICAL,
            "text_reviews_count": ColType.NUMERICAL,
            "publication_year": ColType.NUMERICAL,
            "publication_month": ColType.NUMERICAL,
            "publication_day": ColType.NUMERICAL,
            "publisher": ColType.CATEGORICAL,
            "num_pages": ColType.NUMERICAL,
            "language_code": ColType.CATEGORICAL,
            "format": ColType.CATEGORICAL,
            "isbn": ColType.CATEGORICAL,
            "isbn13": ColType.CATEGORICAL,
            "is_ebook": ColType.CATEGORICAL,
            "kindle_asin": ColType.CATEGORICAL,
            "author_ids": ColType.CATEGORICAL,
            "similar_books": ColType.CATEGORICAL,
            "description": ColType.CATEGORICAL,
        }
        augmented_col_types = {
            **amazon_col_types,
            "book_id": ColType.CATEGORICAL,
            "goodreads_title": ColType.CATEGORICAL,
            "title_without_series": ColType.CATEGORICAL,
            "goodreads_average_rating": ColType.NUMERICAL,
            "ratings_count": ColType.NUMERICAL,
            "text_reviews_count": ColType.NUMERICAL,
            "publication_year": ColType.NUMERICAL,
            "publication_month": ColType.NUMERICAL,
            "publication_day": ColType.NUMERICAL,
            "publisher": ColType.CATEGORICAL,
            "num_pages": ColType.NUMERICAL,
            "language_code": ColType.CATEGORICAL,
            "format": ColType.CATEGORICAL,
            "isbn": ColType.CATEGORICAL,
            "isbn13": ColType.CATEGORICAL,
            "is_ebook": ColType.CATEGORICAL,
            "kindle_asin": ColType.CATEGORICAL,
            "author_ids": ColType.CATEGORICAL,
            "similar_books": ColType.CATEGORICAL,
            "goodreads_description": ColType.CATEGORICAL,
        }

        amazon_df = pd.read_csv(self.raw_paths[0], low_memory=False)
        masks = torch.load(self.raw_paths[4], weights_only=False)
        TableData(
            df=amazon_df,
            col_types=amazon_col_types,
            target_col="categories",
            train_mask=masks["train_mask"],
            val_mask=masks["val_mask"],
            test_mask=masks["test_mask"],
        ).save(self.processed_paths[0])

        goodreads_df = pd.read_csv(self.raw_paths[1], low_memory=False)
        TableData(
            df=goodreads_df,
            col_types=goodreads_col_types,
            target_col=None,
        ).save(self.processed_paths[1])

        agbooks_da_df = pd.read_csv(self.raw_paths[2], low_memory=False)
        masks = torch.load(self.raw_paths[5], weights_only=False)
        TableData(
            df=agbooks_da_df,
            col_types=augmented_col_types,
            target_col="categories",
            train_mask=masks["train_mask"],
            val_mask=masks["val_mask"],
            test_mask=masks["test_mask"],
        ).save(self.processed_paths[2])

        agbooks_fa_df = pd.read_csv(self.raw_paths[3], low_memory=False)
        masks = torch.load(self.raw_paths[4], weights_only=False)
        TableData(
            df=agbooks_fa_df,
            col_types=augmented_col_types,
            target_col="categories",
            train_mask=masks["train_mask"],
            val_mask=masks["val_mask"],
            test_mask=masks["test_mask"],
        ).save(self.processed_paths[3])

    def download(self):
        os.makedirs(self.raw_dir, exist_ok=True)
        path = download_url(self.url, self.raw_dir, "agbooks.zip")
        extract_zip(path, self.raw_dir)
        os.remove(path)

    def __len__(self):
        return 4

    def __getitem__(self, index: int):
        if index < 0 or index >= len(self.data_list):
            raise IndexError
        return self.data_list[index]
