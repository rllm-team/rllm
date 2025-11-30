import os
import os.path as osp
from typing import Optional, Dict, Any
import json

import torch
import pandas as pd
import pyarrow as pa
from pyarrow import parquet as pq

from rllm.data.graph_data import HeteroGraphData
from rllm.types import ColType, TableType
from rllm.data.table_data import TableData
from rllm.datasets.relbench.base import (
    RelBenchDataset,
    RelBenchTableMeta,
    RelBenchTaskType,
    RelBenchTask
)
from rllm.utils.type_infer import TypeInferencer
from rllm.datasets.relbench.utils import (
    save_coltypes,
    save_table_stats,
    upto,
    load_task_data,
    GloveTextEmbedding
)
from rllm.preprocessing import TextEmbedderConfig


class RelF1Dataset(RelBenchDataset):
    """
    A wrapper for rel-f1 dataset in RelBench benchmark from
    `RelBench: A Benchmark for Deep Learning on
    Relational Databases <https://arxiv.org/abs/2407.20060>`__ paper,
    which contains Formula 1 racing data with 9 tables and 3 tasks.

    Tables:
        - circuits
        - constructor_results
        - constructors
        - constructor_standings
        - drivers
        - qualifying
        - races
        - results
        - standings

    Tasks:
        - driver-dnf: Binary classification task to
            predict whether a driver did not finish a race.
        - driver-position: Regression task to
            predict the finishing position of a driver.
        - driver-top3: Binary classification task to
            predict whether a driver finished in the top 3.
    """

    url = "https://relbench.stanford.edu/download/rel-f1/"

    val_timestamp = pd.Timestamp("2005-01-01")
    test_timestamp = pd.Timestamp("2010-01-01")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text_embedder_config = TextEmbedderConfig(
        text_embedder=GloveTextEmbedding(device=device),
        batch_size=256
    )

    def __init__(
        self,
        cached_dir: str,
        force_reload: Optional[bool] = False
    ):
        self.name = "rel-f1"
        root = os.path.join(cached_dir, self.name)
        super().__init__(root, force_reload=force_reload)

    @property
    def tasks(self):
        return ["driver-dnf", "driver-position", "driver-top3"]

    @property
    def table_names(self):
        return [
            "circuits",
            "constructor_results",
            "constructors",
            "constructor_standings",
            "drivers",
            "qualifying",
            "races",
            "results",
            "standings"
        ]

    def process(self):
        r"""
        process data and save to './cached_dir/{dataset}/processed/'.
        """
        os.makedirs(self.processed_dir, exist_ok=True)
        print("Processing raw data, this may take a while...")

        # 1. load parquet files
        print("Loading parquet files...")
        table_df_dict = {}
        table_meta_dict = {}
        for raw_file in self.raw_filenames:
            table_name = raw_file.removesuffix(".parquet")
            path = osp.join(self.db_dir, raw_file)
            table = pq.read_table(path)
            df = table.to_pandas()
            metadata_bytes = table.schema.metadata
            metadata = RelBenchTableMeta(
                fkey_col_to_pkey_table=json.loads(metadata_bytes[b"fkey_col_to_pkey_table"].decode("utf-8")),
                pkey_col=json.loads(metadata_bytes[b"pkey_col"].decode("utf-8")),
                time_col=json.loads(metadata_bytes[b"time_col"].decode("utf-8")) if b"time_col" in metadata_bytes else None
            )

            table_df_dict[table_name] = df
            table_meta_dict[table_name] = metadata

        self._table_meta_dict = table_meta_dict

        # 2. extrat coltype and cache
        print("Inferring column types...")
        table_df_coltype_dict = TypeInferencer.infer_table_df_dict_coltype(
            df_dict=table_df_dict
        )
        save_coltypes(
            table_df_coltype_dict,
            osp.join(self.processed_dir, "coltypes.json")
        )

        # 3. convert to TableData (lazy feature)
        print("Converting to TableData...")
        table_data_dict = {}
        for table_name, df in table_df_dict.items():
            col_types = table_df_coltype_dict[table_name]
            metadata: RelBenchTableMeta = table_meta_dict[table_name]
            table_type = TableType.DATATABLE

            table_data = TableData(
                name=table_name,
                df=df,
                col_types=col_types,
                table_type=table_type,
                pkey=metadata.pkey_col,
                fkeys=metadata.fkey_col_to_pkey_table.keys(),
                time_col=metadata.time_col,
                lazy_feature=True,
            )
            # table_data = upto(table_data, self.test_timestamp)
            table_data_dict[table_name] = table_data

        self._table_dict = table_data_dict

        # 4. validate dataset
        print("Validating dataset...")
        self.validate_dataset()

        # 5. make pkey-fkey graph and cache
        print("Making pkey-fkey graph...")
        hdata, tabledata_stats_dict = self.make_pkey_fkey_graph()
        hdata.save(osp.join(self.processed_dir, "pkey_fkey_graph.pt"))
        save_table_stats(
            tabledata_stats_dict,
            osp.join(self.processed_dir, "tabledata_stats.json")
        )

        self._hdata = hdata
        self._tabledata_stats_dict = tabledata_stats_dict

        # 6. construct tasks
        self._task_dict = {}
        # driver-dnf
        driver_dnf_task = RelBenchTask(
            task_name="driver-dnf",
            task_type=RelBenchTaskType.BINARY_CLASSIFICATION,
            entity_col="driverId",
            entity_table="drivers",
            time_col="date",
            target_col="did_not_finish",
            timedelta=pd.Timedelta(days=30),
            num_eval_timestamps=40,
            task_data_dict=load_task_data(
                task_path=osp.join(self.task_dir, "driver-dnf")
            )
        )
        self._task_dict["driver-dnf"] = driver_dnf_task
        # driver-position
        driver_position_task = RelBenchTask(
            task_name="driver-position",
            task_type=RelBenchTaskType.REGRESSION,
            entity_col="driverId",
            entity_table="drivers",
            time_col="date",
            target_col="position",
            timedelta=pd.Timedelta(days=60),
            num_eval_timestamps=40,
            task_data_dict=load_task_data(
                task_path=osp.join(self.task_dir, "driver-position")
            )
        )
        self._task_dict["driver-position"] = driver_position_task
        # driver-top3
        driver_top3_task = RelBenchTask(
            task_name="driver-top3",
            task_type=RelBenchTaskType.BINARY_CLASSIFICATION,
            entity_col="driverId",
            entity_table="drivers",
            time_col="date",
            target_col="qualifying",
            timedelta=pd.Timedelta(days=30),
            num_eval_timestamps=40,
            task_data_dict=load_task_data(
                task_path=osp.join(self.task_dir, "driver-top3")
            )
        )
        self._task_dict["driver-top3"] = driver_top3_task

        print("Processing done.")

    def __len__(self):
        return 0

    def __getitem__(self, idx):
        pass
