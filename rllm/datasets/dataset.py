from typing import Optional
from abc import ABC, abstractmethod
import os.path as osp

import torch


class Dataset(torch.utils.data.Dataset, ABC):
    r"""An abstract class for creating graph and table datasets.

    Args:
        root (str): Root directory where the dataset should be saved.
        forced_reload (bool): If set to `True`, this dataset will
            be re-process again.
    """

    def __init__(self, root: str, force_reload: Optional[bool] = False):
        self.root = root
        self.raw_dir = osp.join(self.root, "raw")
        self.processed_dir = osp.join(self.root, "processed")
        self.force_reload = force_reload

        if not self.has_download:
            self.download()
        if not self.has_process:
            self.process()

    @property
    def raw_filenames(self):
        r"""file names in the self.raw_dir"""
        raise NotImplementedError

    @property
    def raw_paths(self):
        r"""absolute paths for raw files"""
        return [osp.join(self.raw_dir, file) for file in self.raw_filenames]

    @property
    def processed_filenames(self):
        r"""file names in the self.processed_dir"""
        raise NotImplementedError

    @property
    def processed_paths(self):
        r"""absolute paths for processed files"""
        return [osp.join(self.processed_dir, file) for file in self.processed_filenames]

    @property
    def has_download(self):
        r"""check whether data has been downloaded"""
        return all(
            osp.exists(osp.join(self.raw_dir, file)) for file in self.raw_filenames
        )

    @property
    def has_process(self):
        r"""check whether data has been processed"""
        file_exist = all(
            osp.exists(osp.join(self.processed_dir, file))
            for file in self.processed_filenames
        )
        return file_exist and not self.force_reload

    @abstractmethod
    def download(self):
        r"""download the datasets to self.raw_dir"""
        raise NotImplementedError

    @abstractmethod
    def process(self):
        r"""process the datasets to self.processed_dir"""
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        r"""return the number of data objects"""
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, index: int):
        r"""Gets the data object at index"""
        raise NotImplementedError
