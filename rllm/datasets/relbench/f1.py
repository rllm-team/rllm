import os
import os.path as osp
from typing import Optional, Dict, Any
import json

import torch
import pandas as pd
import pyarrow as pa
from pyarrow import parquet as pq

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
        # self.data_list = []

    @property
    def tasks(self):
        return ["driver-dnf", "driver-position", "driver-top3"]

    @property
    def raw_zip_files(self):
        return [
            "db.zip",
            "tasks/driver-dnf.zip",
            "tasks/driver-position.zip",
            "tasks/driver-top3.zip"
        ]

    @property
    def raw_filenames(self):
        return [
            "circuits.parquet",
            "constructor_results.parquet",
            "constructors.parquet",
            "constructor_standings.parquet",
            "drivers.parquet",
            "qualifying.parquet",
            "races.parquet",
            "results.parquet",
            "standings.parquet",
        ]

    @property
    def processed_filenames(self):
        pass

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

        # 2. extrat coltype
        print("Inferring column types...")
        table_df_coltype_dict = TypeInferencer.infer_table_df_dict_coltype(
            df_dict=table_df_dict
        )
        with open(self.coltypes_path, "w") as f:
            json.dump(
                {
                    table_name: {
                        col_name: col_type.value
                        for col_name, col_type in coltype_dict.items()
                    }
                    for table_name, coltype_dict in table_df_coltype_dict.items()
                },
                f,
                indent=2
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

        # 5. make pkey-fkey graph
        print("Making pkey-fkey graph...")
        self.make_pkey_fkey_graph()

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

    @property
    def task_dict(self) -> Dict[str, RelBenchTask]:
        return self._task_dict

    @property
    def table_dict(self) -> Dict[str, TableData]:
        return self._table_dict

    @property
    def table_meta_dict(self) -> Dict[str, RelBenchTableMeta]:
        return self._table_meta_dict

    @property
    def has_process(self):
        return False

    def __len__(self):
        return 0

    def __getitem__(self, idx):
        pass
