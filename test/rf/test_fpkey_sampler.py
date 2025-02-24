import sys
from os import path as osp
from typing import List
from time import perf_counter

import torch

sys.path.append("./")
from rllm.data.table_data import TableData
from rllm.datasets import TML1MDataset, TACM12KDataset, TLF2KDataset
from rllm.rf.relationframe import RelationFrame, Relation
from rllm.rf.sampler.fpkey_sampler import FPkeySampler

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


def batch_generator(len_target_entry):
    for start in range(0, len_target_entry, 64):
        end = min(start + 64, len_target_entry)
        yield list(range(start, end))


def test_tml1m():
    tables = load_tables('tml1m')
    user_table = tables[0]
    rf = RelationFrame(tables)
    my_fpkey_sampler = FPkeySampler(rf, tables[0])
    train_mask, _, _ = (
        user_table.train_mask,
        user_table.val_mask,
        user_table.test_mask,
    )
    tik = perf_counter()
    for i, batch in enumerate(batch_generator(len(train_mask))):
        new_rf = my_fpkey_sampler.sample(batch)
        print(f"""====> Batch {i},
              table {new_rf.tables[0].table_name} has {len(new_rf.tables[0].df)} entries,
              table {new_rf.tables[1].table_name} has {len(new_rf.tables[1].df)} entries,
              table {new_rf.tables[2].table_name} has {len(new_rf.tables[2].df)} entries""")
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
    print(rf.meta_graph)
    # my_fpkey_sampler = FPkeySampler(rf, tables[0])
    # train_mask, _, _ = (
    #     papers_table.train_mask,
    #     papers_table.val_mask,
    #     papers_table.test_mask,
    # )
    # tik = perf_counter()
    # for i, batch in enumerate(batch_generator(len(train_mask))):
    #     new_rf = my_fpkey_sampler.sample(batch)
    #     print(f"""====> Batch {i},
    #           table {new_rf.tables[0].table_name} has {len(new_rf.tables[0].df)} entries,
    #           table {new_rf.tables[1].table_name} has {len(new_rf.tables[1].df)} entries,
    #           table {new_rf.tables[2].table_name} has {len(new_rf.tables[2].df)} entries,
    #           table {new_rf.tables[3].table_name} has {len(new_rf.tables[3].df)} entries""")
    # tok = perf_counter()
    # print(f"===Total sampling time: {tok-tik :.4} s===")


test_tacm12k()
