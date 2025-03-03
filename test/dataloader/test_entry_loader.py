"""
# test_tml1m()  # 0.08559s, 4 batches
# test_tacm12k()  # 0.07748s, 8 batches
# test_tlf2k()  # 0.4241s, 6 batches
"""
import sys
from os import path as osp
from typing import List
from time import perf_counter

import torch
import pandas as pd

sys.path.append("./")
from rllm.data.table_data import TableData
from rllm.datasets import TML1MDataset, TACM12KDataset, TLF2KDataset
from rllm.data.relationframe import RelationFrame, Relation
from rllm.sampler.fpkey_sampler import FPkeySampler
from rllm.dataloader.entry_loader import EntryLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data")
sjtu_tables = ['tml1m', 'tacm12k', 'tlf2k']


def load_tables(set_name: str) -> List[TableData]:
    if set_name == 'tml1m':
        dataset = TML1MDataset(cached_dir=path, force_reload=True)
        user_table, movie_table, rating_table, _ = dataset.data_list
        rating_table.table_name = "rating_table"
        user_table.table_name = "user_table"
        movie_table.table_name = "movie_table"
        return [user_table, rating_table, movie_table]
    elif set_name == 'tacm12k':
        dataset = TACM12KDataset(cached_dir=path, force_reload=True)
        (
            papers_table,
            authors_table,
            citations_table,
            writings_table,
            _,
            _,
        ) = dataset.data_list
        papers_table.table_name = "papers_table"
        authors_table.table_name = "authors_table"
        citations_table.table_name = "citations_table"
        writings_table.table_name = "writings_table"
        return [papers_table, authors_table, citations_table, writings_table]
    elif set_name == 'tlf2k':
        dataset = TLF2KDataset(cached_dir=path, force_reload=True)
        artist_table, user_artists_table, user_friends_table = dataset.data_list
        artist_table.table_name = "artist_table"
        user_artists_table.table_name = "user_artists_table"
        user_friends_table.table_name = "user_friends_table"
        return [artist_table, user_artists_table, user_friends_table]


def test_tml1m():
    tables = load_tables('tml1m')
    user_table = tables[0]
    rf = RelationFrame(tables)
    dataloader = EntryLoader(
        user_table,
        user_table.train_mask,
        sampling=True,
        rf=rf,
        Sampler=FPkeySampler,
        batch_size=32
    )
    tik = perf_counter()
    for i, batch in enumerate(dataloader):
        print(f"""====> Batch {i},
              table {batch.tables[0].table_name} has {len(batch.tables[0].df)} entries,
              table {batch.tables[1].table_name} has {len(batch.tables[1].df)} entries,
              table {batch.tables[2].table_name} has {len(batch.tables[2].df)} entries""")
    tok = perf_counter()
    print(f"===Total sampling time: {tok-tik :.4} s===")


def test_tacm12k():
    tables = load_tables('tacm12k')
    papers_table = tables[0]
    authors_table = tables[1]
    citations_table = tables[2]
    writings_table = tables[3]
    rel1 = Relation(fkey_table=citations_table, fkey="paper_id", pkey_table=papers_table, pkey="paper_id")
    rel2 = Relation(fkey_table=citations_table, fkey="paper_id_cited", pkey_table=papers_table, pkey="paper_id")
    rel3 = Relation(fkey_table=writings_table, fkey="paper_id", pkey_table=papers_table, pkey="paper_id")
    rel4 = Relation(fkey_table=writings_table, fkey="author_id", pkey_table=authors_table, pkey="author_id")
    rel_l = [rel1, rel2, rel3, rel4]
    rf = RelationFrame(tables, relations=rel_l)

    tik = perf_counter()
    dataloader = EntryLoader(
        papers_table,
        papers_table.train_mask,
        sampling=True,
        rf=rf,
        Sampler=FPkeySampler,
        batch_size=32
    )
    for i, batch in enumerate(dataloader):
        print(f"""====> Batch {i},
              table {batch.tables[0].table_name} has {len(batch.tables[0].df)} entries,
              table {batch.tables[1].table_name} has {len(batch.tables[1].df)} entries,
              table {batch.tables[2].table_name} has {len(batch.tables[2].df)} entries,
              table {batch.tables[3].table_name} has {len(batch.tables[3].df)} entries""")
    tok = perf_counter()
    print(f"===Total sampling time: {tok-tik :.4} s===")


def test_tlf2k():
    def virtual_user_table(n_users: int) -> TableData:
        user_table = TableData(
            df=pd.DataFrame(
                {
                    "userID": [i for i in range(1, n_users + 1)]
                },
            ),
            col_types={},
        )
        user_table.df.set_index('userID', inplace=True)
        user_table.table_name = "user_table"
        return user_table

    tables = load_tables('tlf2k')
    artist_table = tables[0]
    artist_table.df.set_index('artistID', inplace=True)
    user_artists_table = tables[1]
    user_friends_table = tables[2]
    n_users = len(user_artists_table.df['userID'].unique())
    user_table = virtual_user_table(n_users)
    tables.append(user_table)

    rel1 = Relation(fkey_table=user_artists_table, fkey="artistID", pkey_table=artist_table, pkey="artistID")
    rel2 = Relation(fkey_table=user_artists_table, fkey="userID", pkey_table=user_table, pkey="userID")
    rel3 = Relation(fkey_table=user_friends_table, fkey="userID", pkey_table=user_table, pkey="userID")
    rel4 = Relation(fkey_table=user_friends_table, fkey="friendID", pkey_table=user_table, pkey="userID")
    rel_l = [rel1, rel2, rel3, rel4]

    rf = RelationFrame(tables, relations=rel_l)

    dataloader = EntryLoader(
        artist_table,
        artist_table.train_mask,
        sampling=True,
        rf=rf,
        Sampler=FPkeySampler,
        batch_size=32
    )
    tik = perf_counter()
    for i, batch in enumerate(dataloader):
        print(f"""====> Batch {i},
              table {batch.tables[0].table_name} has {len(batch.tables[0].df)} entries,
              table {batch.tables[1].table_name} has {len(batch.tables[1].df)} entries,
              table {batch.tables[2].table_name} has {len(batch.tables[2].df)} entries""")
    tok = perf_counter()
    print(f"===Total sampling time: {tok-tik :.4} s===")
