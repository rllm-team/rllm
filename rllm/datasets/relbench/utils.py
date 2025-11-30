from typing import Dict, Optional, List, Tuple
import json

import torch
import pandas as pd
import pyarrow as pa
from pyarrow import parquet as pq

from rllm.data import TableData
from rllm.datasets.relbench.base import RelBenchTask, RelBenchTableMeta
from rllm.types import ColType, StatType


class GloveTextEmbedding:
    def __init__(self, device: Optional[torch.device] = None):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(
                "sentence-transformers/average_word_embeddings_glove.6B.300d",
                device=device,
            )
        except Exception as e:
            raise ImportError(
                "The 'sentence-transformers' package is required for GloveTextEmbedding. "
                "Install it with 'pip install sentence-transformers'."
            ) from e

    def __call__(self, sentences: List[str]) -> torch.Tensor:
        return self.model.encode(sentences, convert_to_tensor=True)


def upto(table: TableData, timestamp: pd.Timestamp) -> TableData:
    r"""Return a table with all rows upto timestamp (inclusive).

    Table without time_col are returned as is.
    """
    timestamp

    if table.time_col is None:
        return table

    return TableData(
        df=table.df.query(f"{table.time_col} <= @timestamp"),
        col_types=table.col_types,
        table_type=table.table_type,
        pkey=table.pkey,
        fkeys=table.fkeys,
        time_col=table.time_col,
        lazy_feature=True,
    )


def load_task_data(task_path: str) -> Dict[str, Tuple[pd.DataFrame, RelBenchTableMeta]]:

    # load task table
    task_tables = {}
    for split in ["train", "val", "test"]:
        split_file = f"{task_path}/{split}.parquet"
        table = pq.read_table(split_file)
        df = table.to_pandas()
        metadata_bytes = table.schema.metadata
        metadata = RelBenchTableMeta(
            fkey_col_to_pkey_table=json.loads(metadata_bytes[b"fkey_col_to_pkey_table"].decode("utf-8")),
            pkey_col=json.loads(metadata_bytes[b"pkey_col"].decode("utf-8")),
            time_col=json.loads(metadata_bytes[b"time_col"].decode("utf-8")) if b"time_col" in metadata_bytes else None
        )

        input_cols = [
            metadata.time_col,
            *metadata.fkey_col_to_pkey_table.keys(),
        ]

        df = df[input_cols]
        task_tables[split] = (df, metadata)

    return task_tables


def save_coltypes(table_df_coltype_dict: Dict[str, Dict], save_path: str):
    with open(save_path, "w") as f:
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


def load_coltypes(load_path: str) -> Dict[str, Dict]:
    with open(load_path, "r") as f:
        raw_dict = json.load(f)
    return {
        table_name: {
            col_name: ColType(col_type_str)
            for col_name, col_type_str in coltype_dict.items()
        }
        for table_name, coltype_dict in raw_dict.items()
    }


def save_table_stats(table_stats: Dict[str, Dict], save_path: str):
    with open(save_path, "w") as f:
        json.dump(
            {
                table_name: {
                    coltype.value: [
                        {
                            stat_type.value: stat_value
                        }
                        for col_stats in stats_list
                            for stat_type, stat_value in col_stats.items()
                    ]
                    for coltype, stats_list in stats_dict.items()
                }
                for table_name, stats_dict in table_stats.items()
            },
            f,
            indent=2
        )


def load_table_stats(load_path: str) -> Dict[str, Dict]:
    with open(load_path, "r") as f:
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