import sys
from collections import deque
from typing import Any, List, Iterable, Tuple

import numpy as np

sys.path.append("./")
from rllm.data.table_data import TableData
from rllm.rf.relationframe import RelationFrame, Relation
from rllm.rf.sampler.base import BaseSampler, Block


class FPkeySampler(BaseSampler):
    r"""
    fpkey_sampler samples from `seed_table`via the fkey-pkey relation in `rf`.
    """
    def __init__(
        self,
        rf: RelationFrame,
        seed_table: TableData,
        **kwargs
    ):
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

    # 先全采好，然后拼接；结构和原来一样
    def sample(self, index: Iterable) -> Any:
        """
        Samples from the seed_table.
        """
        seed_index = np.array(self.seed_table.df.index[index], dtype=np.int64)

        if len(self._f_p_path) == 0:
            Warning("Only seed_table, no need to sample.")
            sampled_table_data = self.seed_table[seed_index]
            return sampled_table_data, None

        blocks = []
        # seed_table = self.seed_table[seed_index]
        # sampled_tables = [seed_table]
        # cur_table = seed_table
        cur_table = self.seed_table
        cur_src_index = seed_index

        for src_table, dst_table, rel in self._f_p_path:
            """
            rel: fkey_table.fkey ----> pkey_table.pkey
            """
            if src_table is not cur_table:
                # update cur_src_index
                cur_table = src_table
                cur_src_index = np.empty((0,), dtype=np.int64)
                for b in blocks:
                    if b.rel.fkey_table is cur_table:
                        cur_src_index = np.concatenate((cur_src_index, b.src_nodes), axis=0)
                    elif b.rel.pkey_table is cur_table:
                        cur_src_index = np.concatenate((cur_src_index, b.dst_nodes), axis=0)
                
            if src_table is rel.fkey_table:
                """
                src_table.fkey ----> dst_table.pkey
                1 - 1
                """
                dst_index = np.array(src_table.fkey_index(rel.fkey)[cur_src_index], dtype=np.int64)

        # for sample_depth, item in enumerate(self._f_p_path):
        #     """
        #     rel: fkey_table.fkey_col ----> pkey_table.pkey_col
        #     """
        #     src_table, dst_table, rel = item
        #     # if sample_depth == 0:
        #     #     p_index = seed_index
            
        #     if src_table is not cur_table:
        #         sampled_src_table = []
        #         for t in sampled_tables:
        #             if t.table_name == src_table.table_name:
        #                 sampled_src_table.append(t)

        #     if rel.fkey_table is src_table:
        #         """
        #         1 - 1
        #         For each src entry, sample one dst entry. O(k)
        #         """
        #         dst_index = np.array(src_table.fkey_index(rel.fkey)[p_index], dtype=np.int64)
        #         blocks.append(Block(edge_list=np.stack([p_index, dst_index], axis=1), rel=rel))
        #         dst_table = dst_table[dst_index]
        #         sampled_tables.append(dst_table)
                
        #     elif rel.pkey_table is src_table:
        #         """
        #         1 - n
        #         For each src entry, sample all dst entries. O(k*n)
        #         """
        #         edges = np.empty((0, 2), dtype=np.int64)
        #         for i in p_index:
        #             cur = np.where(dst_table.fkey_index(rel.fkey) == i)[0]
        #             cur = np.stack([cur, np.full_like(cur, i, dtype=np.int64)], axis=1)
        #             edges = np.concatenate((edges, cur), axis=0)
        #         blocks.append(Block(edge_list=edges, rel=rel))
        #         sampled_tables.append(dst_table[edges[:, 0]])
        #         # self._f_p_path[i][1] = dst_table
        #     else:
        #         raise ValueError("Invalid: Src_table: {}, Dst_table: {}, Rel: {}".format(
        #             src_table.table_name,
        #             dst_table.table_name,
        #             rel))

        return RelationFrame(tables=sampled_tables), blocks
