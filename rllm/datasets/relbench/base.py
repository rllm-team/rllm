import os
import os.path as osp
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
import warnings
from enum import Enum

import tqdm
import numpy as np
import pandas as pd

from rllm.types import ColType, TableType
from rllm.data import TableData, HeteroGraphData
from rllm.datasets.dataset import Dataset
from rllm.utils.download import download_url
from rllm.utils.extract import extract_zip


@dataclass
class RelBenchTableMeta:
    fkey_col_to_pkey_table: Dict[str, str]
    pkey_col: str
    time_col: Optional[str] = None


class RelBenchTaskType(Enum):
    REGRESSION = "regression"
    BINARY_CLASSIFICATION = "binary_classification"
    # MULTICLASS_CLASSIFICATION = "multiclass_classification"
    # MULTILABEL_CLASSIFICATION = "multilabel_classification"


@dataclass
class RelBenchTask:
    task_name: str
    task_type: RelBenchTaskType
    entity_col: str
    entity_table: str
    time_col: str
    target_col: str
    timedelta: pd.Timedelta
    num_eval_timestamps: int

    # split: ["train", "val", "test"] data
    task_data_dict: Dict[str, Tuple[pd.DataFrame, RelBenchTableMeta]]


class RelBenchDataset(Dataset):
    """
    Override methods for RelBench datasets.
    """
    @property
    def db_dir(self):
        return osp.join(self.raw_dir, "db")

    @property
    def task_dir(self):
        return osp.join(self.raw_dir, "tasks")

    @property
    def coltypes_path(self):
        return osp.join(self.processed_dir, "coltypes.json")

    @property
    def has_download(self):
        for db_file in self.raw_filenames:
            if not osp.exists(osp.join(self.db_dir, db_file)):
                return False
        for subtask in self.tasks:
            sub_task_dir =  osp.join(self.task_dir, subtask)
            for split in ["train", "val", "test"]:
                split_file = osp.join(sub_task_dir, f"{split}.parquet")
                if not osp.exists(split_file):
                    return False
        return True

    def download(self):
        """
        Download and unzip raw files.
        """
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.task_dir, exist_ok=True)
        print("Downloading raw files...")
        print(self.task_dir)
        for filename in self.raw_zip_files:
            url = self.url + filename
            path = download_url(url, self.raw_dir, filename)
            if filename.startswith("tasks/"):
                extract_zip(path, self.task_dir)
            else:
                # extract db files
                extract_zip(path, self.raw_dir)
            os.remove(path)

    def validate_dataset(self):
        """
        Validate the integrity of downloaded files.
        1. validate primary keys
        2. validate foreign keys (correct if necessary)
        """
        # 1. validate primary keys
        for table_name, table in self.table_dict.items():
            if table.pkey is not None:
                ser = table.df.index
                if not (ser.values == np.arange(len(ser))).all():
                    raise ValueError(
                        f"Primary key column {table.pkey} in table {table_name} is not valid."
                    )

        # 2. validate foreign keys
        for table_name, table in self.table_dict.items():
            metadata: RelBenchTableMeta = self.table_meta_dict[table_name]
            for fkey_col, pkey_table_name in metadata.fkey_col_to_pkey_table.items():
                pkey_range = len(self.table_dict[pkey_table_name].df)
                mask = table.df[fkey_col] >= pkey_range
                if mask.any():
                    warnings.warn(
                        f"Foreign key column {fkey_col} in table {table_name} has values over {pkey_range}. "
                        f"Correcting them by setting to None."
                    )
                    table.df.loc[mask, fkey_col] = None

    def make_pkey_fkey_graph(self) -> HeteroGraphData:
        """
        Make primary key - foreign key graph for the dataset.
        Cached to self.processed_dir/pkey_fkey_graph.pt
        """
        hdata = HeteroGraphData()
        col_stats_dict = {}

        table_dict = self.table_dict

        for table_name, table in tqdm.tqdm(table_dict.items(), desc="Processing tables"):
            df = table.df

            # Ensure that pkey is consecutive.
            if table.pkey is not None:
                assert (df.index.values == np.arange(len(df))).all()

            col_to_coltype = table.col_types

            # remove pkey, fkeys in col_to_coltype
            self._remove_pkey_fkeys(col_to_coltype, table)

            # add constant feature in case df is empty:
            if len(col_to_coltype) == 0:
                col_to_coltype = {"__const__": ColType.NUMERICAL}
                # We need to add edges later, so we need to also keep the fkeys
                fkey_dict = {key: df[key] for key in table.fkeys}
                df = pd.DataFrame({"__const__": np.ones(len(table.df)), **fkey_dict})

            # tensorize and cache
            cache_path = osp.join(self.processed_dir, f"{table_name}.pt")

            print(f"Lazy materializing table {table_name}...")
            table.lazy_materialize(
                keep_df=True,
                text_embedder_config=getattr(self, "text_embedder_config", None),
            )

            print(table.feat_dict.keys())
            for key, val in table.feat_dict.items():
                print(f"  {key}: {val.shape}")
                print(val[:3])

            table.save(cache_path)
            hdata[table_name] = table

        print(hdata)
        graph_cache_path = osp.join(self.processed_dir, "pkey_fkey_graph.pt")
        hdata.save(graph_cache_path)
        exit()


    # private methods
    def _remove_pkey_fkeys(self, col_to_type: Dict[str, Any], table: TableData):
        """Inplace remove pkey and fkeys from col_to_type."""
        if table.pkey is not None:
            if table.pkey in col_to_type:
                col_to_type.pop(table.pkey)
        for fkey in table.fkeys:
            if fkey in col_to_type:
                col_to_type.pop(fkey)

    # Additional required abstract methods
    @property
    def task_dict(self) -> Dict[str, RelBenchTask]:
        raise NotImplementedError

    @property
    def table_dict(self) -> Dict[str, TableData]:
        raise NotImplementedError

    @property
    def table_meta_dict(self) -> Dict[str, RelBenchTableMeta]:
        raise NotImplementedError

    @property
    def tasks(self) -> List[str]:
        raise NotImplementedError

    @property
    def raw_zip_files(self) -> List[str]:
        raise NotImplementedError

    # Placeholder implementations
    def process(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError
