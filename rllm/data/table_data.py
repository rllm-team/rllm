from __future__ import annotations
import gc
from functools import lru_cache, wraps
from typing import Any, Dict, List, Union, Tuple, Callable, Optional, Sequence, overload
from uuid import uuid4
from warnings import warn
import copy

import torch
import numpy as np
from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from rllm.types import ColType, TaskType, StatType, TableType
from rllm.data.storage import BaseStorage


class BaseTable:
    r"""An abstract base class for table data storage."""

    @classmethod
    def load(cls, path: str):
        data = torch.load(path, weights_only=False)
        return cls(**data)

    def save(self, path: str):
        torch.save(self.to_dict(), path)

    def to_dict(self):
        return self.__dict__

    def apply(self, func: Callable, *args: str):
        raise NotImplementedError

    def to(self, device: Union[int, str], *args: str, non_blocking: bool = False):
        return self.apply(
            lambda x: x.to(device=device, non_blocking=non_blocking), *args
        )

    def cpu(self, *args: str):
        return self.apply(lambda x: x.cpu(), *args)

    def cuda(
        self,
        device: Optional[Union[int, str]] = None,
        *args: str,
        non_blocking: bool = False,
    ):
        device = "cuda" if device is None else device
        return self.apply(lambda x: x.cuda(device, non_blocking=non_blocking), *args)

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
        return {key: tensor[idx] for key, tensor in self.feat_dict.items()}, self.y[idx]


class TableData(BaseTable):
    r"""A base class for creating single table data.

    TableData is designed with lazy feature generation in mind.
    Call `lazy_materialize` to materialize `feat_dict` and `metadata`.

    TableData always unify `df.index` as pkey for eazy process.
    If `pkey` is `None`, use `cls.NONEPKEY` instead.

    TableData use `BaseStorage` to store normal properties like `df`, `pkey`.
    Extra private properties should be listed in `cls.PRIVATE_PROPERTIES`.

    Args:
        df (DataFrame): The tabular data frame containing the dataset.
        col_types (Dict[str, ColType]): A dictionary mapping each column
            in the data frame to a semantic type (e.g., CATEGORICAL, NUMERICAL).
        name (str, optional): The name of the table. If None, use `table_`
            + `uuid4` instead.
            (default: :obj:`None`)
        table_type (TableType, optional): The type of the table.
            (default: :obj:`None`, in which case it will be inferred)
        pkey (str, optional): The column name used as the primary key
            for the table.
            (default: :obj:`None`, in which case use `df.index.name`)
        fkey_to_table_map (Dict[str, str], optional): A dictionary mapping
            foreign keys to the tables they reference.
            (default: :obj:`None`, in which case it will be inferred)
        lazy_feature (bool, optional): Whether to generate features lazily.
            If set to :obj:`True`, features will only be generated by called
            `lazy_materialize` method.
            (default: :obj:`False` for compatibility)
        feat_dict (Dict[ColType, Tensor], optional): A dictionary storing
            tensors for each column type
            (default: :obj:`None`, in which case it will be generated)
        metadata (Dict[ColType, List[dict[str, Any]]], optional):
            Metadata for each column type, specifying the statistics and
            properties of the columns.
            (default: :obj:`None`)
        target_col (str, optional): The column name used as the target for
            prediction tasks.
            (default: :obj:`None`)
        y (Tensor, optional): A tensor containing the target values.
            (default: :obj:`None`, in which case it will be generated)
        **kwargs: Additional key-value attributes to set as instance variables.
    """

    NONEPKEY = '_NonePkey'
    PRIVATE_PROPERTIES = ['_mapping', '_fkeys', '_inherit_feat_dict']

    def __init__(
        self,
        # base data table
        df: DataFrame,
        col_types: Dict[str, ColType],
        # additional attributes
        name: Optional[str] = None,
        table_type: Optional[TableType] = None,
        pkey: Optional[str] = None,
        fkeys: Optional[Sequence[str]] = None,
        # lazy_feature
        lazy_feature: bool = False,
        feat_dict: Dict[ColType, Tensor] = None,
        metadata: Dict[ColType, List[dict[str, Any]]] | None = None,
        # task table
        target_col: Optional[str] = None,
        y: Tensor = None,
        **kwargs,
    ):
        self._mapping = BaseStorage()

        # base
        self.df = df
        self.col_types = col_types

        # additional
        self.table_name = name or "table_" + str(uuid4())
        self.pkey = pkey if pkey is not None else df.index.name
        self.fkeys_ = list(fkeys) if fkeys is not None else None
        self.table_type = table_type or self.infer_table_type()

        self._unify_valid_pkey()

        # task table
        self.target_col = target_col
        self.y = y

        for key, value in kwargs.items():
            setattr(self, key, value)

        self._validate()

        # lazy feature generation
        self.feat_dict = feat_dict
        self.metadata = metadata

        if feat_dict is not None:
            self._inherit_feat_dict = True
        else:
            if lazy_feature:
                self._inherit_feat_dict = False
            else:
                self._generate_feat_dict()
                self._inherit_feat_dict = False

        if metadata is None:
            if lazy_feature:
                pass
            else:
                self._generate_metadata()

    # init funcs ##########################################
    def infer_table_type(self) -> TableType:
        r"""Infer the table type.
        Tend to infer as a data table,
        unless table has no primary key and multiple foreign keys.
        This func may not be accruate, please check the result.
        """
        if self.pkey is None:
            if (self.fkeys_ is not None and
                    len(self.fkeys_) > 1):
                return TableType.RELATIONSHIPTABLE
        return TableType.DATATABLE

    def _unify_valid_pkey(self):
        r"""Reindex self.df to make sure `pkey` is index column.
        1. validate that `pkey` is existing and unique;
        2. unify `pkey` as `df.index`.
        """
        # pkey column is not index -> set_index(self.pkey)
        if self.pkey != self.df.index.name:
            assert self.pkey in self.df.columns, (
                f"Pkey {self.pkey} is not in `df`.")

            p_col = self.df[self.pkey]
            assert p_col.nunique == len(p_col), (
                f"Pkey {self.pkey} column is not unique."
            )

            self.df.set_index(self.pkey, inplace=True, drop=True)
        # pkey column is index and `None`
        elif self.pkey is None:
            self.df.index.name = self.NONEPKEY
            self.pkey = self.NONEPKEY

    def _validate(self):
        r"""Validate the table data."""
        assert self.df is not None, "Dataframe is not provided."
        assert self.col_types is not None, "Column types are not provided."
        if self.fkeys_ is not None:
            for fkey in self.fkeys_:
                assert fkey in self.df.columns, f"{fkey} is not in the dataframe."

    # lazy feature generation ###########################
    def lazy_materialize(self, keep_df: bool = True):
        r"""Materialize the `feat_dict` and `metadata`.

        Args:
            keep_df (bool, optional): Whether to keep the raw dataframe.
                (default: :obj:`True`)
        """
        self._generate_feat_dict()
        self._generate_metadata()
        self._len = next(iter(self.feat_dict.values())).size(0)
        if not keep_df:
            self.df = None  # remove df
            gc.collect()

    def if_materialized(self):
        return self.feat_dict is not None

    def after_materialize(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if self.feat_dict is None:
                raise ValueError(f"Function `{func.__name__}()` requires feat_dict,",
                                 "but it is not generated yet.")
            return func(self, *args, **kwargs)
        return wrapper

    def df_requisite(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if self.df is None:
                raise ValueError(f"Function `{func.__name__}()` requires dataframe,",
                                 "but it is removed after materialized.")
            return func(self, *args, **kwargs)
        return wrapper

    # base functions #####################################
    @classmethod
    def load(cls, path: str) -> TableData:
        key_map = {
            "table_name": "name",
            "_fkeys": "fkeys",
        }
        data = torch.load(path, weights_only=False)
        for old_key, new_key in key_map.items():
            if old_key in data.keys():
                data[new_key] = data.pop(old_key)
        # TODO: Delete this
        # key_mapping = {"get_split_func": "get_split"}

        # for old_key, new_key in key_mapping.items():
        #     if old_key in data.keys():
        #         data[new_key] = data.pop(old_key)

        return cls(**data)

    def to_dict(self):
        return self._mapping.to_dict()

    def apply(self, func: Callable, *args: str):
        self._mapping.apply(func, *args)
        return self

    def __getattr__(self, key: str):
        # avoid infinite loop.
        if key == "_mapping":
            self.__dict__["_mapping"] = BaseStorage()
            return self.__dict__["_mapping"]

        return getattr(self._mapping, key)

    def __setattr__(self, key: str, value: Any):
        propobj = getattr(self.__class__, key, None)
        if propobj is not None and getattr(propobj, "fset", None) is not None:
            propobj.fset(self, value)
        elif key[:1] == "_":
            self.__dict__[key] = value
        else:
            setattr(self._mapping, key, value)

    def __delattr__(self, key: str):
        if key[:1] == "_":
            del self.__dict__[key]
        else:
            del self[key]

    def __repr__(self) -> str:
        return (
            f"TableData(name={self.table_name}, \n"
            f"  table_type={self.table_type}) \n"
        )

    @lru_cache
    def __len__(self) -> int:
        if hasattr(self, "_len"):  # after materialized
            return self._len
        return len(self.df)

    @overload
    def __getitem__(self, index: ColType) -> Tensor:
        # Return feat_dict tensor of certain column type.
        ...

    @overload
    def __getitem__(self, index: Tensor) -> TableData:
        # Return a new TableData with choiced data.
        ...

    @after_materialize
    def __getitem__(self, index: Union[ColType, Tensor]) -> Any:
        """TODO: ColType; Return df and tensor simultaneously."""
        if isinstance(index, ColType):
            # Each ColType consists many column,
            # ordered by col_types given in init function.
            assert index in self.col_types.values()
            return self.feat_dict[index]

        # Support Tensor slicing for hetero graph laoder.
        # TableData saved as node attribute.
        if isinstance(index, Tensor):
            return self._tensor_slice(index)

    # base properties ####################################
    @property
    def fkeys(self) -> List[str]:
        r"""The foreign keys of the table."""
        return self.fkeys_ if self.fkeys_ is not None else []

    @fkeys.setter
    @df_requisite
    def fkeys(self, fkeys: List[str]):
        for fkey in fkeys:
            assert fkey in self.df.columns, f"{fkey} is not in the dataframe."
        self.fkeys_ = fkeys

    @property
    def index_col(self) -> Optional[str]:
        r"""The name of the index column.
        TableData always uses `pkey` as `df.index.name`
        """
        return self.pkey

    @lru_cache
    @df_requisite
    def fkey_index(self, fkey_col: str) -> np.ndarray:
        r"""fkey_index for sampler."""
        return self.df[fkey_col].values

    @property
    @df_requisite
    def cols(self) -> List[str]:
        r"""The columns of the table data, including index and target columns."""
        return [self.pkey] + list(self.df.columns)

    @property
    def feat_cols(self) -> List[str]:
        r"""The input feature columns of the dataset."""
        cols = list(self.col_types.keys())
        if self.target_col is not None:
            cols.remove(self.target_col)
        for fkey in self.fkeys_:
            if fkey in cols:
                cols.remove(fkey)
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
        return self.__len__()

    @property
    def num_cols(self):
        r"""The number of feat columns we used."""
        return len(self.feat_cols)

    @property
    @df_requisite
    def num_classes(self) -> Optional[int]:
        # assert self.target_col is not None
        # ZK: If none, return None, but raise error. I change this for sampling.
        if self.target_col is None:
            return None
        num_classes = self.df[self.target_col].nunique()
        assert num_classes > 1
        return num_classes

    def count_numerical_features(self) -> int:
        r"""Return numerical features"""
        numerics = []
        for col_name, col_type in self.col_types.items():
            if col_type == ColType.NUMERICAL:
                numerics.append(col_name)

        return numerics

    @df_requisite
    def count_categorical_features(self) -> dict[str, int]:
        r"""Return categorical features and its count of unique values"""
        categories = {}
        for col_name, col_type in self.col_types.items():
            if col_type == ColType.CATEGORICAL and col_name != self.target_col:
                categories[col_name] = self.df[col_name].nunique()

        return categories

    # after materialize ##################################
    @after_materialize
    def get_feat_dict(
        self, start: int | float = 0.0, end: int | float = 1.0
    ) -> dict[ColType, Tensor]:
        assert isinstance(
            start, type(end)
        ), "`start` and `end` must \
            be same type! \
            Integers correspond to actual rows, \
            while floats correspond to proportions."
        if isinstance(start, int):
            start_row = start
            end_row = end
        if isinstance(start, float):
            assert (
                start >= 0 and end <= 1
            ), "when start and end are ratios, \
                they must be between 0 and 1!"
            start_row = int(round(self.num_rows * start))
            end_row = int(start + round(self.num_rows * (end - start)))
        feat_dict = {}
        # Get tensors corresponding to each in ColType
        for col_type in self.feat_dict.keys():
            feat_dict[col_type] = self.feat_dict[col_type][start_row:end_row]
        return feat_dict

    @after_materialize
    def get_feat_dict_from_mask(self, mask: Tensor) -> dict[ColType, Tensor]:
        feat_dict = {}
        for col_type in self.feat_dict.keys():
            feat_dict[col_type] = self.feat_dict[col_type][mask]
        return feat_dict

    @after_materialize
    @df_requisite
    def shuffle(self, return_perm: bool = False):
        perm = torch.randperm(len(self))
        self.df = self.df.iloc[perm].reset_index(drop=True)
        for col_type in self.feat_dict.keys():
            self.feat_dict[col_type] = self.feat_dict[col_type][perm]
        self.y = self.y[perm]
        if return_perm:
            return perm

    @after_materialize
    def get_dataset(
        self,
        train_split: int | float,
        val_split: int | float,
        test_split: int | float,
    ) -> Tuple[TableDataset, TableDataset, TableDataset]:
        assert isinstance(train_split, type(val_split)) and isinstance(
            val_split, type(test_split)
        ), "train_split, val_split and test_split must besame type! \
                Integers correspond to actual rows, \
                while floats correspond to proportions."
        if isinstance(train_split, float):
            assert (
                abs(train_split + val_split + test_split - 1.0) < 1e-9
            ), "train, val and test ratio must sum up to 1.0!"
            train_split = round(self.num_rows * train_split)
            val_split = train_split + round(self.num_rows * val_split)
        else:
            # assert(train_split + val_split + test_split) == self.num_rows, \
            # "train, val, and test rows must sum up to total rows!"
            val_split = train_split + val_split
        return (
            TableDataset(self.get_feat_dict(0, train_split), self.y[:train_split]),
            TableDataset(
                self.get_feat_dict(train_split, val_split),
                self.y[train_split:val_split],
            ),
            TableDataset(
                self.get_feat_dict(val_split, self.num_rows), self.y[val_split:]
            ),
        )

    @after_materialize
    def get_dataset_from_mask(
        self,
        train_mask: Tensor,
        val_mask: Tensor,
        test_mask: Tensor,
    ) -> Tuple[TableDataset, TableDataset, TableDataset]:
        return (
            TableDataset(self.get_feat_dict_from_mask(train_mask), self.y[train_mask]),
            TableDataset(self.get_feat_dict_from_mask(val_mask), self.y[val_mask]),
            TableDataset(self.get_feat_dict_from_mask(test_mask), self.y[test_mask]),
        )

    @after_materialize
    def get_dataloader(
        self,
        train_split: int | float,
        val_split: int | float,
        test_split: int | float,
        batch_size: int,
        shuffle: bool = False,
    ) -> Tuple[TableDataset, TableDataset, TableDataset]:
        train_dataset, val_dataset, test_dataset = self.get_dataset(
            train_split, val_split, test_split
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
        return train_loader, val_loader, test_loader

    # Materialize functions, get table tensor ##################
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
            feat_dict[ColType.CATEGORICAL] = feat_dict[ColType.CATEGORICAL].int()
        self.feat_dict = feat_dict

    @df_requisite
    def _generate_column_tensor(self, col: str = None):
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

    def _generate_metadata(
        self,
    ):
        r"""Get each column's statistical data from single tabular dataset.
        Columns with same ColType will be integrated together.
        eg: {ColType.CATEGORICAL: [{col_name: col_name1, stat1: xx, stat2: xx},
        {col_name: col_name2, stat1: xx, stat2: xx}], ...}"""
        metadata = {}
        # 1. Iterate each column
        col_types = self.col_types.copy()
        col_types.pop(self.target_col, None)
        col_types_count = {}
        for col_name, col_type in col_types.items():
            sub_stats_list = {}

            if col_type not in metadata.keys():
                # add a new list for certain ColType
                metadata[col_type] = []

            # 2. Compute stats
            stats_to_compute = StatType.stats_for_col_type(col_type)
            if col_type not in col_types_count.keys():
                col_types_count[col_type] = 0
            current_col_index = col_types_count[col_type]
            col_types_count[col_type] = col_types_count[col_type] + 1
            for stat_type in stats_to_compute:
                sub_stats_list[stat_type] = StatType.compute(
                    self.feat_dict[col_type][:, current_col_index], stat_type
                )

            # 3. Update metadata
            sub_stats_list[StatType.COLNAME] = col_name
            metadata[col_type].append(sub_stats_list)

        self.metadata = metadata
        return self

    # Helper functions ####################################
    def __copy__(self) -> TableData:
        out = self.__class__.__new__(self.__class__)
        for k, v in self.__dict__.items():
            out.__dict__[k] = v
        out.__dict__["_mapping"] = copy.copy(self._mapping)
        out._mapping._parent = out
        return out

    def _tensor_slice(self, index: Tensor) -> TableData:
        r"""Tensor slice only apply to feat_dict, and y
        i.e. slice materialized Tensor data.
        Other attributes like `df`, `metadata` are shallow copied
        to save memory.

        Warning:
            This function keeps `metadata` original, rather than re-calculate.
        """
        if not self.if_materialized():
            warn("Tensor slicing on a non-materialized TableData.")

        assert index.dim() == 1, f"Index should be 1-D, but {index.dim()} found."

        out = copy.copy(self)
        out.feat_dict = {}
        for ctype in self.feat_dict.keys():
            out.feat_dict[ctype] = self.feat_dict[ctype][index]

        out.y = self.y[index]

        out.__dict__['_len'] = index.numel()

        return out
