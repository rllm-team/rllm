import json
import os
import os.path as osp
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
import warnings
from enum import Enum
import tqdm

import torch
import numpy as np
import pandas as pd

from rllm.types import ColType, StatType
from rllm.data import TableData, HeteroGraphData
from rllm.datasets.dataset import Dataset
from rllm.utils import download_url, extract_zip, sort_edge_index
from rllm.utils.col_process import timecol_to_unix_time


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

    def save(self, save_path: str):
        assert save_path.endswith(".pt")
        torch.save(self, save_path)

    @staticmethod
    def load(save_path: str) -> "RelBenchTask":
        assert save_path.endswith(".pt")
        task = torch.load(save_path, weights_only=False)
        return task


class RelBenchDataset(Dataset):
    """
    Override methods for RelBench datasets.

    Subclasses need to assign the following properties after processing:
        self._task_dict: Dict[str, RelBenchTask]
        self._table_dict: Dict[str, TableData]
        self._hdata: HeteroGraphData
        self._tabledata_stats_dict: Dict[str, Any]
        self._table_meta_dict: Dict[str, RelBenchTableMeta]
    """

    COLTYPE_FILE = "coltypes.json"
    HDATA_FILE = "pkey_fkey_graph.pt"
    TABLEDATA_STATS_FILE = "tabledata_stats.json"
    TABLE_META_FILE = "table_meta.json"

    ###############################################################
    # abstract properties which need to be implemented by subclasses
    url = None

    val_timestamp: Optional[pd.Timestamp] = None
    test_timestamp: Optional[pd.Timestamp] = None

    @property
    def tasks(self) -> List[str]:
        raise NotImplementedError

    @property
    def table_names(self) -> List[str]:
        raise NotImplementedError

    # placeholder implementations for abstract methods
    def process(self):
        raise NotImplementedError

    #################################################################
    # interface properties and methods
    @property
    def raw_zip_files(self) -> List[str]:
        return ["db.zip"] + [f"tasks/{task}.zip" for task in self.tasks]

    @property
    def raw_filenames(self):
        return [
            f"{table_name}.parquet" for table_name in self.table_names
        ]

    @property
    def processed_filenames(self):
        return (
            [f"{table_name}.pt" for table_name in self.table_names]
            + [f"{task_name}.pt" for task_name in self.tasks]
            + [self.TABLE_META_FILE]
            + [self.COLTYPE_FILE]
            + [self.HDATA_FILE]
            + [self.TABLEDATA_STATS_FILE]
        )

    # path properties
    @property
    def db_dir(self):
        return osp.join(self.raw_dir, "db")

    @property
    def task_dir(self):
        return osp.join(self.raw_dir, "tasks")

    # after process properties
    @property
    def coltypes(self) -> Dict[str, Dict]:
        if not hasattr(self, "_coltypes"):
            self._coltypes = self._try_load_coltypes()
        return self._coltypes

    @property
    def table_dict(self) -> Dict[str, TableData]:
        if not hasattr(self, "_table_dict"):
            self._table_dict = self._try_load_cached_table_dict()
        return self._table_dict

    @property
    def table_meta_dict(self) -> Dict[str, RelBenchTableMeta]:
        if not hasattr(self, "_table_meta_dict"):
            self._table_meta_dict = self._try_load_cached_table_meta_dict()
        return self._table_meta_dict

    @property
    def hdata(self) -> HeteroGraphData:
        if not hasattr(self, "_hdata"):
            self._hdata = self._try_load_hdata()
        return self._hdata

    @property
    def tabledata_stats_dict(self) -> Dict[str, Any]:
        if not hasattr(self, "_tabledata_stats_dict"):
            self._tabledata_stats_dict = self._try_load_tabledata_stats_dict()
        return self._tabledata_stats_dict

    @property
    def task_dict(self) -> Dict[str, RelBenchTask]:
        if not hasattr(self, "_task_dict"):
            self._task_dict = self._try_load_task_dict()
        return self._task_dict

    def load_all(self):
        """Force load all cached properties."""
        if not self.has_process:
            raise ValueError("Dataset has not been processed yet.")
        _ = self.coltypes
        _ = self.table_dict
        _ = self.table_meta_dict
        _ = self.hdata
        _ = self.tabledata_stats_dict
        _ = self.task_dict

    #####################################################################
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
        print("All raw files are present.")
        return True

    @property
    def has_process(self):
        file_exist = all(
            osp.exists(osp.join(self.processed_dir, file))
            for file in self.processed_filenames
        )
        return file_exist

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

    def make_pkey_fkey_graph(self) -> Tuple[HeteroGraphData, Dict]:
        """
        Make primary key - foreign key graph for the dataset.

        This method lazy materializes each TableData, saves them to processed_dir,
        and constructs the HeteroGraphData based on pkey-fkey relations.

        Returns:
            HeteroGraphData: Heterogeneous graph data.
            Dict: table_name -> TableData.metadata
        """
        hdata = HeteroGraphData()
        tabledata_stats_dict = {}   # table_name -> metadata

        table_dict = self.table_dict
        table_meta_dict = self.table_meta_dict

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

            table.save(cache_path)

            # Add table data to hetero graph data
            hdata[table_name].table = table
            if table.time_col is not None:
                hdata[table_name].time = torch.from_numpy(
                    timecol_to_unix_time(table.df[table.time_col])
                )

            # Add table column stats
            tabledata_stats_dict[table_name] = table.metadata

            # Add edges based on pkey-fkey relations
            for fkey_col_name, pkey_table_name in (
                table_meta_dict[table_name].fkey_col_to_pkey_table.items()
            ):
                pkey_index = df[fkey_col_name]  # pkey be referenced by fkey
                # Filter out dangling foreign keys
                mask = ~pkey_index.isna()
                fkey_index = torch.arange(len(pkey_index))

                pkey_index = torch.from_numpy(
                    pkey_index[mask].astype(int).values
                )
                fkey_index = fkey_index[torch.from_numpy(mask.values)]
                # Ensure no dangling fkeys
                assert (pkey_index < len(table_dict[pkey_table_name].df)).all()

                # fkey -> pkey edges (this table -> pkey_table)
                edge_index = torch.stack(
                    [fkey_index, pkey_index], dim=0
                )
                edge_type = (table_name, f"f2p_{fkey_col_name}", pkey_table_name)
                hdata[edge_type].edge_index = sort_edge_index(edge_index)

                # pkey -> fkey edges (pkey_table -> this table)
                # add "rev_" as revserse edge (used for undirected graph)
                edge_index = torch.stack(
                    [pkey_index, fkey_index], dim=0
                )
                edge_type = (pkey_table_name, f"rev_f2p_{fkey_col_name}", table_name)
                hdata[edge_type].edge_index = sort_edge_index(edge_index)

        if  hdata.validate():
            print("HeteroGraphData validation passed.")
        else:
            print("HeteroGraphData validation failed.")

        return hdata, tabledata_stats_dict


    # private methods
    def _remove_pkey_fkeys(self, col_to_type: Dict[str, Any], table: TableData):
        """Inplace remove pkey and fkeys from col_to_type."""
        if table.pkey is not None:
            if table.pkey in col_to_type:
                col_to_type.pop(table.pkey)
        for fkey in table.fkeys:
            if fkey in col_to_type:
                col_to_type.pop(fkey)

    def _save_coltypes(self, coltypes: Dict[str, Dict]):
        coltype_path = osp.join(self.processed_dir, self.COLTYPE_FILE)
        with open(coltype_path, "w") as f:
            json.dump(
                {
                    table_name: {
                        col_name: col_type.value
                        for col_name, col_type in coltype_dict.items()
                    }
                    for table_name, coltype_dict in coltypes.items()
                },
                f,
                indent=2
            )

    def _try_load_coltypes(self) -> Dict[str, Dict]:
        if not self.has_process:
            raise ValueError("Dataset has not been processed yet.")
        coltype_path = osp.join(self.processed_dir, self.COLTYPE_FILE)
        with open(coltype_path, "r") as f:
            raw_dict = json.load(f)
        return {
            table_name: {
                col_name: ColType(col_type_str)
                for col_name, col_type_str in coltype_dict.items()
            }
            for table_name, coltype_dict in raw_dict.items()
        }

    def _try_load_cached_table_dict(self) -> Dict[str, TableData]:
        """Load cached TableData from processed_dir."""
        if not self.has_process:
            raise ValueError("Dataset has not been processed yet.")
        table_dict = {}
        for table_name in self.table_names:
            cache_path = osp.join(self.processed_dir, f"{table_name}.pt")
            table_data = TableData.load(cache_path)
            table_dict[table_name] = table_data
        return table_dict

    def _save_table_meta_dict(self, table_meta_dict: Dict[str, RelBenchTableMeta]):
        """Save table_meta_dict to processed_dir."""
        meta_path = osp.join(self.processed_dir, self.TABLE_META_FILE)
        serializable_dict = {}
        for table_name, meta in table_meta_dict.items():
            serializable_dict[table_name] = {
                "fkey_col_to_pkey_table": meta.fkey_col_to_pkey_table,
                "pkey_col": meta.pkey_col,
                "time_col": meta.time_col,
            }
        with open(meta_path, "w") as f:
            json.dump(serializable_dict, f)

    def _try_load_cached_table_meta_dict(self) -> Dict[str, RelBenchTableMeta]:
        """Load cached table_meta_dict from processed_dir."""
        if not self.has_process:
            raise ValueError("Dataset has not been processed yet.")
        meta_path = osp.join(self.processed_dir, self.TABLE_META_FILE)
        with open(meta_path, "r") as f:
            raw_dict = json.load(f)
        table_meta_dict = {}
        for table_name, meta_dict in raw_dict.items():
            table_meta_dict[table_name] = RelBenchTableMeta(
                fkey_col_to_pkey_table=meta_dict["fkey_col_to_pkey_table"],
                pkey_col=meta_dict["pkey_col"],
                time_col=meta_dict["time_col"],
            )
        return table_meta_dict

    def _try_load_hdata(self) -> HeteroGraphData:
        """Load cached HeteroGraphData from processed_dir."""
        if not self.has_process:
            raise ValueError("Dataset has not been processed yet.")
        hdata_path = osp.join(self.processed_dir, self.HDATA_FILE)
        hdata = HeteroGraphData.load(hdata_path)
        return hdata

    def _save_tabledata_stats_dict(self, table_stats: Dict[str, Any]):
        stats_path = osp.join(self.processed_dir, self.TABLEDATA_STATS_FILE)
        with open(stats_path, "w") as f:
            res = {}
            for table_name, stats_dict in table_stats.items():
                res[table_name] = {}
                for coltype, stats_list in stats_dict.items():
                    res[table_name][coltype.value] = []
                    for col_stats in stats_list:
                        stat_entry = {}
                        for stat_type, stat_value in col_stats.items():
                            stat_entry[stat_type.value] = stat_value
                        res[table_name][coltype.value].append(stat_entry)
            json.dump(res, f, indent=2)

    def _try_load_tabledata_stats_dict(self) -> Dict[str, Any]:
        """Load cached tabledata_stats_dict from processed_dir."""
        if not self.has_process:
            raise ValueError("Dataset has not been processed yet.")
        stats_path = osp.join(self.processed_dir, self.TABLEDATA_STATS_FILE)
        with open(stats_path, "r") as f:
            raw_dict = json.load(f)

        res = {}
        for table_name, stats_dict in raw_dict.items():
            res[table_name] = {}
            for coltype_str, stats_list in stats_dict.items():
                coltype = ColType(coltype_str)
                res[table_name][coltype] = []
                for stat_entry in stats_list:
                    stat_converted = {}
                    for stat_type_str, stat_value in stat_entry.items():
                        stat_type = StatType(stat_type_str)
                        stat_converted[stat_type] = stat_value
                    res[table_name][coltype].append(stat_converted)
        return res

    def _save_task_dict(self, task_dict: Dict[str, RelBenchTask]):
        """Save task_dict to processed_dir."""
        for task_name, task in task_dict.items():
            task_path = osp.join(self.processed_dir, f"{task_name}.pt")
            task.save(task_path)

    def _try_load_task_dict(self) -> Dict[str, RelBenchTask]:
        """Load cached task_dict from processed_dir."""
        if not self.has_process:
            raise ValueError("Dataset has not been processed yet.")
        task_dict = {}
        for task_name in self.tasks:
            task_path = osp.join(self.processed_dir, f"{task_name}.pt")
            task = RelBenchTask.load(task_path)
            task_dict[task_name] = task
        return task_dict

    # override other methods
    def __len__(self):
        return len(self.table_names)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.table_names):
            raise IndexError
        table_name = self.table_names[idx]
        return self._table_dict[table_name]
