from __future__ import annotations

from collections.abc import Iterable
from typing import (Any, Dict, Union, Tuple, Callable, Optional)

import torch
import numpy as np
from pandas import DataFrame
from torch import Tensor
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder

from rllm.data.storage import BaseStorage
from rllm.types import ColType, TaskType, StatType


class BaseTable:
    r"""An abstract base class for table data storage."""
    @classmethod
    def load(cls, path: str):
        data = torch.load(path)
        return cls(**data)

    def save(self, path: str):
        torch.save(self.to_dict(), path)

    def to_dict(self):
        return self.__dict__

    def apply(self, func: Callable, *args: str):
        raise NotImplementedError

    def to(self, device: Union[int, str], *args: str,
           non_blocking: bool = False):
        return self.apply(
            lambda x: x.to(device=device, non_blocking=non_blocking), *args)

    def cpu(self, *args: str):
        return self.apply(lambda x: x.cpu(), *args)

    def cuda(self, device: Optional[Union[int, str]] = None, *args: str,
             non_blocking: bool = False):
        device = 'cuda' if device is None else device
        return self.apply(lambda x: x.cuda(device, non_blocking=non_blocking),
                          *args)

    def pin_memory(self, *args: str):
        return self.apply(lambda x: x.pin_memory(), *args)


class TableDataset(Dataset):
    r"""Table dataset inherited from :class:`torch.utils.data.Dataset`"""
    def __init__(self, feat_dict, y):
        self.feat_dict = feat_dict
        self.y = y

    def __len__(self):
        return self.y.size(0)

    def __getitem__(self, idx):
        return {
            key: tensor[idx]
            for key, tensor in self.feat_dict.items()
        }, self.y[idx]


class TableData(BaseTable):
    r"""A base class for creating single table data.

    Args:
        df (DataFrame): The tabular data frame.
        col_types (Dict[str, ColType]): A dictionary that maps
            each column in the data frame to a semantic type.
        target_col (str, optional): The column used as target.
            (default: :obj:`None`)
    """
    def __init__(
        self,
        df: DataFrame,
        col_types: Dict[str, ColType],
        target_col: str | None = None,
        # TODO: The following variables should not be explicitly defined
        feat_dict: Dict[ColType, Tensor] = None,
        y: Tensor = None,
        stats_dict: dict[ColType, list[dict[str, Any]]] | None = None,
        **kwargs,
    ):
        self._mapping = BaseStorage()

        self.df = df
        self.stats_dict = stats_dict
        self.target_col = target_col
        self.col_types = col_types
        self.feat_dict = feat_dict
        self.y = y

        for key, value in kwargs.items():
            setattr(self, key, value)

        if feat_dict is None or y is None:
            self._generate_feat_dict()
        if stats_dict is None:
            self._generate_stats_dict()

    @classmethod
    def load(cls, path: str) -> TableData:
        data = torch.load(path)
        # TODO: Delete this
        key_mapping = {'get_split_func': 'get_split'}

        for old_key, new_key in key_mapping.items():
            if old_key in data.keys():
                data[new_key] = data.pop(old_key)

        return cls(**data)

    def to_dict(self):
        return self._mapping.to_dict()

    def apply(self, func: Callable, *args: str):
        self._mapping.apply(func, *args)
        return self

    def __getattr__(self, key: str):
        # avoid infinite loop.
        if key == '_mapping':
            self.__dict__['_mapping'] = BaseStorage()
            return self.__dict__['_mapping']

        return getattr(self._mapping, key)

    def __setattr__(self, key: str, value: Any):
        propobj = getattr(self.__class__, key, None)
        if propobj is not None and getattr(propobj, 'fset', None) is not None:
            propobj.fset(self, value)
        elif key[:1] == '_':
            self.__dict__[key] = value
        else:
            setattr(self._mapping, key, value)

    def __delattr__(self, key: str):
        if key[:1] == '_':
            del self.__dict__[key]
        else:
            del self[key]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: Union[int, Iterable, ColType]) -> Any:
        """TODO: ColType; Return df and tensor simultaneously."""
        if isinstance(index, ColType):
            # Each ColType consists many column,
            # ordered by col_types given in init function.
            assert index in self.col_types.values()
            return self.feat_dict[index]

    @property
    def feat_cols(self) -> list[str]:
        r"""The input feature columns of the dataset."""
        cols = list(self.col_types.keys())
        if self.target_col is not None:
            cols.remove(self.target_col)
        return cols

    @property
    def task_type(self) -> TaskType:
        r"""The task type of the dataset."""
        assert self.target_col is not None
        if self.col_types[self.target_col] == ColType.CATEGORICAL:
            if self.num_classes == 2:
                return TaskType.BINARY_CLASSIFICATION
            else:
                return TaskType.MULTI_CLASSIFICATION
        elif self.col_types[self.target_col] == ColType.NUMERICAL:
            return TaskType.REGRESSION
        else:
            raise ValueError("Task type cannot be inferred.")

    @property
    def num_rows(self):
        r"""The number of rows of the dataset."""
        return len(self.df)

    @property
    def num_cols(self):
        r"""The number of columns we usedt."""
        return len(self.feat_cols)

    @property
    def num_classes(self) -> int:
        assert self.target_col is not None
        num_classes = self.df[self.target_col].nunique()
        assert num_classes > 1
        return num_classes

    def get_feat_dict(
        self,
        start: int | float = 0.0,
        end: int | float = 1.0
    ) -> dict[ColType, Tensor]:
        assert isinstance(start, type(end)), "`start` and `end` must \
            be same type! \
            Integers correspond to actual rows, \
            while floats correspond to proportions."
        if isinstance(start, int):
            start_row = start
            end_row = end
        if isinstance(start, float):
            assert start >= 0 and end <= 1, "when start and end are ratios, \
                they must be between 0 and 1!"
            start_row = int(round(self.num_rows * start))
            end_row = int(start + round(self.num_rows * (end - start)))
        feat_dict = {}
        # Get tensors corresponding to each in ColType
        for col_type in self.feat_dict.keys():
            feat_dict[col_type] = self.feat_dict[col_type][start_row:end_row]
        return feat_dict

    def get_feat_dict_from_mask(
        self,
        mask: Tensor
    ) -> dict[ColType, Tensor]:
        feat_dict = {}
        for col_type in self.feat_dict.keys():
            feat_dict[col_type] = self.feat_dict[col_type][mask]
        return feat_dict

    def count_numerical_features(self) -> int:
        r"""Return numerical features"""
        numerics = []
        for col_name, col_type in self.col_types.items():
            if col_type == ColType.NUMERICAL:
                numerics.append(col_name)

        return numerics

    def count_categorical_features(self) -> dict[str, int]:
        r"""Return categorical features and its count of unique values"""
        categories = {}
        for col_name, col_type in self.col_types.items():
            if col_type == ColType.CATEGORICAL and col_name != self.target_col:
                categories[col_name] = self.df[col_name].nunique()

        return categories

    def shuffle(self, return_perm: bool = False):
        perm = torch.randperm(len(self))
        self.df = self.df.iloc[perm].reset_index(drop=True)
        for col_type in self.feat_dict.keys():
            self.feat_dict[col_type] = self.feat_dict[col_type][perm]
        self.y = self.y[perm]
        if return_perm:
            return perm

    def get_dataset(
        self,
        train_split: int | float,
        val_split: int | float,
        test_split: int | float,
    ) -> Tuple[TableDataset, TableDataset, TableDataset]:
        assert isinstance(train_split, type(val_split)) and \
            isinstance(val_split, type(test_split)), \
            "train_split, val_split and test_split must besame type! \
                Integers correspond to actual rows, \
                while floats correspond to proportions."
        if isinstance(train_split, float):
            assert abs(train_split + val_split + test_split - 1.0) < 1e-9, \
                "train, val and test ratio must sum up to 1.0!"
            train_split = round(self.num_rows * train_split)
            val_split = train_split + round(self.num_rows * val_split)
        else:
            # assert(train_split + val_split + test_split) == self.num_rows, \
            # "train, val, and test rows must sum up to total rows!"
            val_split = train_split + val_split
        return (
            TableDataset(
                self.get_feat_dict(0, train_split),
                self.y[:train_split]
            ),
            TableDataset(
                self.get_feat_dict(train_split, val_split),
                self.y[train_split:val_split]
            ),
            TableDataset(
                self.get_feat_dict(val_split, self.num_rows),
                self.y[val_split:]
            )
        )

    def get_dataset_from_mask(
        self,
        train_mask: Tensor,
        val_mask: Tensor,
        test_mask: Tensor,
    ) -> Tuple[TableDataset, TableDataset, TableDataset]:
        return (
            TableDataset(
                self.get_feat_dict_from_mask(train_mask),
                self.y[train_mask]
            ),
            TableDataset(
                self.get_feat_dict_from_mask(val_mask),
                self.y[val_mask]
            ),
            TableDataset(
                self.get_feat_dict_from_mask(test_mask),
                self.y[test_mask]
            )
        )

    # Get table tensor #########################################
    def _generate_feat_dict(
        self,
    ):
        r"""Get feat dict from single tabular dataset."""
        # 1. Iterate each column
        feat_dict = {}
        for col, col_type in self.col_types.items():
            # 2. Get column tensor
            col_tensor = self._generate_column_tensor(col)

            # 3. Update feat dict
            if col == self.target_col:
                self.y = col_tensor
                continue
            if col_type not in feat_dict.keys():
                feat_dict[col_type] = []
            feat_dict[col_type].append(col_tensor.reshape(-1, 1))

        # 4. Concat column tensors
        for col_type, xs in feat_dict.items():
            feat_dict[col_type] = torch.cat(xs, dim=-1)

        # TODO: Change hard-coding here
        if ColType.CATEGORICAL in feat_dict.keys():
            feat_dict[ColType.CATEGORICAL] = \
                feat_dict[ColType.CATEGORICAL].int()
        self.feat_dict = feat_dict

    def _generate_column_tensor(
        self,
        col: str = None
    ):
        col_types = self.col_types[col]
        col_copy = self.df[col].copy()

        if col_types == ColType.NUMERICAL:
            if col_copy.isnull().any():
                col_copy.fillna(np.nan, inplace=True)

        elif col_types == ColType.CATEGORICAL:
            if col_copy.isnull().any():
                col_copy.fillna(-1, inplace=True)

            col_fit = col_copy[col_copy != -1]
            labels = LabelEncoder().fit_transform(col_fit)
            col_copy[col_copy != -1] = labels

        return torch.tensor(col_copy.values.astype(float), dtype=torch.float32)

    def _generate_stats_dict(
        self,
    ):
        r"""Get each column's statistical data from single tabular dataset.
        Columns with same ColType will be integrated together.
        eg: {ColType.CATEGORICAL: [{col_name: col_name1, stat1: xx, stat2: xx},
        {col_name: col_name2, stat1: xx, stat2: xx}], ...}"""
        stats_dict = {}
        # 1. Iterate each column
        col_types = self.col_types.copy()
        col_types.pop(self.target_col, None)
        col_types_count = {}
        for col_name, col_type in col_types.items():
            sub_stats_list = {}

            if col_type not in stats_dict.keys():
                # add a new list for certain ColType
                stats_dict[col_type] = []

            # 2. Compute stats
            stats_to_compute = StatType.stats_for_col_type(col_type)
            if col_type not in col_types_count.keys():
                col_types_count[col_type] = 0
            current_col_index = col_types_count[col_type]
            col_types_count[col_type] = col_types_count[col_type] + 1
            for stat_type in stats_to_compute:
                sub_stats_list[stat_type] = StatType.compute(
                    self.feat_dict[col_type][:, current_col_index],
                    stat_type
                )

            # 3. Update stats_dict
            sub_stats_list[StatType.COLNAME] = col_name
            stats_dict[col_type].append(sub_stats_list)

        self.stats_dict = stats_dict
        return self
