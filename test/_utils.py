from typing import Tuple
import os.path as osp

from rllm.datasets import TML1MDataset, TACM12KDataset
from rllm.data import TableData, RelationFrame, Fkey2PkeyTable
from rllm.types import TableType


def load_tml1m_tables() -> Tuple[TableData]:
    root = osp.dirname(osp.dirname(__file__))
    data_path = osp.join(root, "data")
    dataset = TML1MDataset(cached_dir=data_path, force_reload=True)
    user_table, movie_table, rating_table, _ = dataset.data_list
    rating_table.table_name = "rating_table"
    user_table.table_name = "user_table"
    movie_table.table_name = "movie_table"
    rating_table.fkeys = ["UserID", "MovieID"]
    rating_table.table_type = TableType.RELATIONSHIPTABLE
    return user_table, movie_table, rating_table


def load_tacm12k_rf() -> RelationFrame:
    root = osp.dirname(osp.dirname(__file__))
    data_path = osp.join(root, "data")
    dataset = TACM12KDataset(cached_dir=data_path, force_reload=True)
    (
        papers_table,
        authors_table,
        citations_table,
        writings_table,
        _,
        _
    ) = dataset.data_list
    papers_table.table_name = 'papers_table'
    authors_table.table_name = 'authors_table'
    citations_table.table_name = 'citations_table'
    writings_table.table_name = 'writings_table'
    citations_table.fkeys = ['paper_id', 'paper_id_cited']
    writings_table.fkeys = ["paper_id", "author_id"]
    fpt_l = [
        Fkey2PkeyTable(
            citations_table.table_name, 'paper_id', papers_table.table_name
        ),
        Fkey2PkeyTable(
            citations_table.table_name, 'paper_id_cited', papers_table.table_name
        ),
        Fkey2PkeyTable(
            writings_table.table_name, 'paper_id', papers_table.table_name
        ),
        Fkey2PkeyTable(
            writings_table.table_name, 'author_id', authors_table.table_name
        ),
    ]
    return RelationFrame(
        tables=[
            papers_table,
            authors_table,
            citations_table,
            writings_table,
        ],
        fkey2pkey_tables=fpt_l
    )
