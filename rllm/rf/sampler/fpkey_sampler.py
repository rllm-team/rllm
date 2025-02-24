import sys
from collections import deque
from typing import List, Iterable, Tuple, Union

import numpy as np
import pandas as pd

sys.path.append("./")
from rllm.data.table_data import TableData
from rllm.rf.relationframe import RelationFrame, Relation, Block
from rllm.rf.sampler.base import BaseSampler


class FPkeySampler(BaseSampler):
    r"""
    fpkey_sampler samples from `seed_table`via the fkey-pkey relation in `rf`.

    Args:
        rf: RelationFrame
        seed_table: TableData
    """
    def __init__(
        self,
        rf: RelationFrame,
        seed_table: TableData,
        **kwargs
    ):
        assert rf.validate_rels()
        rf.unify_f_pkey_dtype()
        self.rf = rf
        assert seed_table in rf.tables, "seed_table should be in rf.tables."
        self.seed_table = seed_table
        self._f_p_path = self._bfs_meta_g()
        super().__init__(**kwargs)

    @property
    def f_p_path(self) -> List[Tuple[TableData, TableData]]:
        r"""Returns the fkey-pkey paths."""
        return self._f_p_path

    def _hierarchical_rels(self):
        r"""Construct hierarchical relations from the seed_table."""
        _f_p_path = []
        q = deque([self.seed_table])
        rels_l: List[Relation] = self.rf.relations.copy()
        while q:
            cur_table = q.popleft()
            cur_path = []
            for rel in rels_l:
                if rel.fkey_table is cur_table:
                    cur_path.append()
                    q.append(rel.pkey_table)
                    rels_l.remove(rel)
                elif rel.pkey_table is cur_table:
                    cur_path.append(rel)
                    q.append(rel.fkey_table)
                    rels_l.remove(rel)
            if cur_path:
                _f_p_path.append(cur_path)
            if not rels_l:
                break

        return _f_p_path

    def _bfs_meta_g(self) -> List[Tuple[TableData, TableData]]:
        # TODO (ZK), use degrees and directed graph BFS instead to take multi-edge graph into account.
        r"""BFS traverse the undirected relationframe meta graph."""
        visited = set()
        queue = deque([self.seed_table])
        res = []

        while queue:
            cur = queue.popleft()
            if cur not in visited:
                visited.add(cur)
                for neigh in self.rf.undirected_meta_graph.neighbors(cur):
                    if neigh not in visited:
                        queue.append(neigh)
                        res.append((cur, neigh, self.rf.undirected_meta_graph.edges[cur, neigh]['relation']))

        assert len(res) == len(self.rf.relations), "The meta graph is not connected."
        return res

    def index_union(self, indexes: Union[List[pd.Index], List[np.ndarray]]) -> pd.Index:
        r"""Union the indexes."""
        if isinstance(indexes[0], np.ndarray):
            indexes = [pd.Index(i) for i in indexes]
        res = indexes[0]
        for i in indexes[1:]:
            res = res.union(i)
        return res

    def merge_tables(self, blocks: List[Block]) -> RelationFrame:
        r"""Merge the blocks."""
        sampled_tables = []

        for table in self.rf.tables:
            cur_index = []
            for block in blocks:
                if block.rel.fkey_table is table:
                    cur_index.append(block.src_nodes)
                elif block.rel.pkey_table is table:
                    cur_index.append(block.dst_nodes)

            cur_index = self.index_union(cur_index)
            new_table = table.sample(cur_index)
            sampled_tables.append(new_table)

        rf = RelationFrame(tables=sampled_tables, _blocks=blocks)
        return rf

    def sample(self, index: Iterable) -> RelationFrame:
        """
        Samples from the seed_table.

        Args:
            index: The sampling index of the seed_table. From 0 to len(seed_table).
        """

        seed_index = self.seed_table.df.index[index]

        if len(self._f_p_path) == 0:
            Warning("Only seed_table, no need to sample.")
            sampled_table_data = self.seed_table[seed_index]
            return sampled_table_data, None

        blocks = []
        cur_table = self.seed_table
        cur_src_index = seed_index

        for src_table, dst_table, rel in self._f_p_path:
            """
            rel: fkey_table.fkey ----> pkey_table.pkey
            """
            if src_table is not cur_table:
                # update cur_src_index
                cur_table = src_table
                cur_src_index = []
                for b in blocks:
                    if b.rel.fkey_table is cur_table:
                        cur_src_index.append(b.src_nodes)
                    elif b.rel.pkey_table is cur_table:
                        cur_src_index.append(b.dst_nodes)
                if cur_src_index:
                    cur_src_index = self.index_union(cur_src_index)
                else:
                    raise ValueError("No blocks are sampled.")

            if src_table is rel.fkey_table:
                """
                src_table.fkey ----> dst_table.pkey
                1 - 1
                """
                dst_index = src_table.fkey_index(rel.fkey)[cur_src_index]
                block = Block(
                    edge_list=(cur_src_index, dst_index),
                    rel=rel,
                    src_nodes_=cur_src_index,
                    dst_nodes_=dst_index
                )
                blocks.append(block)

            elif src_table is rel.pkey_table:
                """
                src_table.pkey ----> dst_table.fkey
                1 - n
                """
                edges = [np.empty((0,), dtype=int), np.empty((0,), dtype=int)]
                for i in cur_src_index:
                    indices = np.where(dst_table.fkey_index(rel.fkey) == i)[0]
                    # trans array indices to df.index
                    cur_fkey_ind = dst_table.df.index[indices]
                    cur_src_edges = np.full((len(cur_fkey_ind),), i)
                    edges[0] = (np.concatenate((edges[0], cur_fkey_ind), axis=0, dtype=cur_fkey_ind.dtype))
                    edges[1] = (np.concatenate((edges[1], cur_src_edges), axis=0, dtype=cur_src_edges.dtype))

                """
                Sampling edge order is reversed to the block edge order.
                Sampling edge order: src_table(fpkey_table).pkey ----> dst_table(fkey_table).fkey
                Block edge order: fkey_table.fkey ----> pkey_table.pkey
                """
                block = Block(
                    edge_list=edges,
                    rel=rel,
                    src_nodes_=edges[0],
                    dst_nodes_=cur_src_index
                )
                blocks.append(block)

            else:
                raise ValueError("Invalid: Src_table: {}, Dst_table: {}, Rel: {}".format(
                    src_table.table_name,
                    dst_table.table_name,
                    rel))

        if blocks:
            return self.merge_tables(blocks)
        else:
            raise ValueError("No blocks are sampled.")
