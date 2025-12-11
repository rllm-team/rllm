from typing import Dict, Optional, List, Tuple
import json

import torch
import pandas as pd
from pyarrow import parquet as pq

from rllm.data import TableData
from rllm.datasets.relbench.base import RelBenchTableMeta


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

        task_tables[split] = (df, metadata)

    return task_tables
