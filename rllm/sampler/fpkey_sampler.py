from collections import deque
from typing import List, Iterable, Tuple, Union

import numpy as np
import pandas as pd
import networkx as nx

from rllm.sampler.base import BaseSampler
from rllm.data import TableData, RelationFrame, Relation, Block


class FPkeySampler(BaseSampler):
    r"""
    fpkey_sampler samples from `seed_table`via the fkey-pkey relation in `rf`.

    seed_table will always be the first table to sample, which keeps training/val/test
    mask can be accessed by ``seed_table.xxx[ : batch_size]``.

    Args:
        rf: RelationFrame
        seed_table: TableData
    """
    def __init__(
        self,
        rf: RelationFrame,
        seed_table: TableData,
        f_p_path: List[Tuple[TableData, TableData, Relation]] = None,
        **kwargs
    ):
        assert rf.validate_rels()
        rf.unify_f_pkey_dtype()
        self.rf = rf
        assert seed_table in rf.tables, "seed_table should be in rf.tables."
        self.seed_table = seed_table

        # sampling order
        if f_p_path is not None:
            self._f_p_path = f_p_path
        else:
            self._f_p_path = self.__bfs_meta_g()

        super().__init__(**kwargs)

    @property
    def f_p_path(self) -> List[Tuple[TableData, TableData, Relation]]:
        r"""Returns the fkey-pkey paths."""
        return self._f_p_path

    def __bfs_meta_g(self) -> List[Tuple[TableData, TableData, Relation]]:
        r"""BFS traverse the relationframe meta graph."""
        meta_g: nx.MultiDiGraph = self.rf.meta_graph
        visited = set()
        queue = deque([self.seed_table])
        res = []
        while queue:
            cur = queue.popleft()
            if cur not in visited:
                visited.add(cur)
                for neigh in meta_g.neighbors(cur):
                    if neigh not in visited:
                        queue.append(neigh)
                        for e_data in meta_g.get_edge_data(cur, neigh).values():
                            res.append((cur, neigh, e_data['relation']))
                for neigh_ in meta_g.predecessors(cur):
                    if neigh_ not in visited:
                        queue.append(neigh_)
                        for e_data in meta_g.get_edge_data(neigh_, cur).values():
                            """
                            Direction here implies the sampling order, but not the relation's direction.
                            """
                            res.append((cur, neigh_, e_data['relation']))

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

        def merge_one(table: TableData) -> TableData:
            r"""Merge one table."""
            cur_index = []
            for block in blocks:
                if block.nrel.fkey_table == table.table_name:
                    cur_index.append(block.src_nodes)
                elif block.nrel.pkey_table == table.table_name:
                    cur_index.append(block.dst_nodes)
            cur_index = self.index_union(cur_index)
            return table.sample(cur_index)

        sampled_tables = [merge_one(self.seed_table)]  # Always put seed_table at first.

        tables = set(self.rf.tables) - {self.seed_table}
        while tables:
            table = tables.pop()
            sampled_tables.append(merge_one(table))

        nrels = [b.nrel for b in blocks]
        rf = RelationFrame(sampled_tables, sampled=True, _blocks=blocks, relations=nrels)
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
        # cur_table = self.seed_table
        cur_src_index = seed_index

        for src_table, dst_table, rel in self._f_p_path:
            """
            rel: fkey_table.fkey ----> pkey_table.pkey
            """

            if src_table is rel.fkey_table:
                """
                src_table.fkey ----> dst_table.pkey
                1 - 1
                """
                dst_index = src_table.fkey_index(rel.fkey)[cur_src_index]
                block = Block(
                    edge_list=(cur_src_index, dst_index),
                    nrel=rel.to_name(),
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
                    nrel=rel.to_name(),
                    src_nodes_=edges[0],
                    dst_nodes_=cur_src_index
                )
                blocks.append(block)

            else:
                raise ValueError("Invalid: Src_table: {}, Dst_table: {}, Rel: {}".format(
                    src_table.table_name,
                    dst_table.table_name,
                    rel))

            # update cur_src_index
            cur_src_index = []
            for block in blocks:
                if block.nrel.pkey_table == dst_table.table_name:
                    cur_src_index.append(block.dst_nodes)
                elif block.nrel.fkey_table == dst_table.table_name:
                    cur_src_index.append(block.src_nodes)
            if cur_src_index:
                cur_src_index = self.index_union(cur_src_index)
            else:
                raise ValueError("No blocks are sampled.")

        if blocks:
            rf = self.merge_tables(blocks)
            setattr(rf, 'target_table_name', self.seed_table.table_name)
            setattr(rf, 'sampling_seeds', list(seed_index))
            return rf
        else:
            raise ValueError("No blocks are sampled.")
