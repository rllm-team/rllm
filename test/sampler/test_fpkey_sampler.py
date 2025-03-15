"""
test_tml1m()  # 0.07824s 4 batches
test_tlf2k()  # 1.679s 6 batches
test_tacm12k()  # 0.01513 8 batches
"""
import sys
from os import path as osp
from typing import List
from time import perf_counter

import torch
import pandas as pd

sys.path.append("./")
from rllm.datasets import TML1MDataset, TACM12KDataset, TLF2KDataset
from rllm.data import TableData, RelationFrame, Relation
from rllm.sampler import FPkeySampler

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


def batch_generator(set):
    for start in range(0, len(set), 32):
        end = min(start + 32, len(set))
        if end - start < 2:
            break
        yield set[start:end]


def test_tml1m():
    tables = load_tables('tml1m')
    user_table = tables[0]
    rf = RelationFrame(tables)
    my_fpkey_sampler = FPkeySampler(rf, tables[0])
    for src, dst, rel in my_fpkey_sampler.f_p_path:
        print(f"sampling order: {src.table_name} ----> {dst.table_name}")
        print("relation:", rel)
    train_mask, _, _ = (
        user_table.train_mask,
        user_table.val_mask,
        user_table.test_mask,
    )
    tik = perf_counter()
    for i, batch in enumerate(batch_generator(torch.nonzero(train_mask).flatten().tolist())):
        new_rf = my_fpkey_sampler.sample(batch)
        print(f"""====> Batch {i}:
              table {new_rf.tables[0].table_name} has {len(new_rf.tables[0])} entries,
              table {new_rf.tables[1].table_name} has {len(new_rf.tables[1])} entries,
              table {new_rf.tables[2].table_name} has {len(new_rf.tables[2])} entries""")
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
    f_p_path = [(papers_table, citations_table, rel1),
                (citations_table, papers_table, rel2),
                (writings_table, papers_table, rel3),
                (writings_table, authors_table, rel4)]
    my_fpkey_sampler = FPkeySampler(rf, tables[0], f_p_path=f_p_path)
    for src, dst, rel in my_fpkey_sampler.f_p_path:
        print(f"sampling order: {src.table_name} ----> {dst.table_name}")
        print("relation:", rel)
    train_mask, _, _ = (
        papers_table.train_mask,
        papers_table.val_mask,
        papers_table.test_mask,
    )
    tik = perf_counter()
    for i, batch in enumerate(batch_generator(torch.nonzero(train_mask).flatten().tolist())):
        new_rf = my_fpkey_sampler.sample(batch)
        print(f"""====> Batch {i},
              table {new_rf.tables[0].table_name} has {len(new_rf.tables[0])} entries,
              table {new_rf.tables[1].table_name} has {len(new_rf.tables[1])} entries,
              table {new_rf.tables[2].table_name} has {len(new_rf.tables[2])} entries,
              table {new_rf.tables[3].table_name} has {len(new_rf.tables[3])} entries""")
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
    f_p_path = [(artist_table, user_artists_table, rel1),
                (user_artists_table, user_table, rel2),
                (user_table, user_friends_table, rel3),
                (user_friends_table, user_table, rel4)]

    rf = RelationFrame(tables, relations=rel_l, f_p_path=f_p_path)
    my_fpkey_sampler = FPkeySampler(rf, artist_table)
    for src, dst, rel in my_fpkey_sampler.f_p_path:
        print(f"sampling order: {src.table_name} ----> {dst.table_name}")
        print("relation:", rel)

    tik = perf_counter()
    for i, batch in enumerate(batch_generator(torch.nonzero(artist_table.train_mask).flatten().tolist())):
        new_rf = my_fpkey_sampler.sample(batch)
        print(f"""====> Batch {i},
              table {new_rf.tables[0].table_name} has {len(new_rf.tables[0])} entries,
              table {new_rf.tables[1].table_name} has {len(new_rf.tables[1])} entries,
              table {new_rf.tables[2].table_name} has {len(new_rf.tables[2])} entries,
              table {new_rf.tables[3].table_name} has {len(new_rf.tables[3])} entries""")
    tok = perf_counter()
    print(f"===Total sampling time: {tok-tik :.4} s===")


# test_tml1m()  # 0.07824s 4 batches
# test_tlf2k()  # 1.679s 6 batches
# test_tacm12k()  # 0.01513 8 batches
