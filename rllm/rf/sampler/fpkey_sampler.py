import sys
from collections import deque
from typing import Any, Dict, List, Literal, Optional, Union, Tuple, Iterable

import torch
from torch import Tensor
import numpy as np
import networkx as nx

sys.path.append("./")
from rllm.data.table_data import TableData, SubTableData
from rllm.data.graph_data import GraphData
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
        # self._hierarchical_rels() # deprecated
        self._f_p_path = self._bfs_meta_g()
        super().__init__(**kwargs)

    @property
    def f_p_paths(self) -> List[TableData]:
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
    
    def _bfs_meta_g(self) -> List[TableData]:
        r"""BFS traverse the undirected relationframe meta graph."""
        visited = set()
        queue = deque([self.seed_table])
        res = []

        while queue:
            cur = queue.popleft()
            if cur not in visited:
                visited.add(cur)
                res.append(cur)
                for neigh in self.rf.undirected_meta_graph.neighbors(cur):
                    if neigh not in visited:
                        queue.append(neigh)
        return res
    
    # def tensorize(self):
    #     r"""Tensorize the fkey-pkey cols to accelerate sample."""
    #     pass

    #TODO：目前还是麻烦了，想办法提前构建rel的边索引，然后直接取就行了
    #TODO：返回值目前还没想好，rf的话有些臃肿，但是方便把方法集成好
    #TODO：或者直接返回edgelist 的 tensor + tables? 那感觉不如返回rf
    #TODO：现在的BRIDGE里为什么TNN只作用于target table？别的table不需要TNN吗？那adj里是啥。。。？
    def sample(self, index: Iterable) -> Any:
        """
        Samples from the seed_table.
        """
        index = np.array(index)
        sampling_depth = len(self.f_p_paths) - 1

        if sampling_depth == 0:
            Warning("Only seed_table, no need to sample.")
            sampled_table_data = self.seed_table[index]
            return sampled_table_data, None
        
        blocks = []
        sampled_tables = [self.seed_table[index]]
        for layer in range(sampling_depth):
            src_table: TableData = self.f_p_paths[layer]
            dst_table: TableData = self.f_p_paths[layer + 1]
            """
            rel: fkey_table.fkey_col ----> pkey_table.pkey_col
            """
            rel: Relation = self.rf.undirected_meta_graph.edges[src_table, dst_table]['relation'] # sampled edge dirc

            if rel.fkey_table is src_table:
                """
                1 - 1
                For each src entry, sample one dst entry. O(k)
                """
                dst_index = np.array(src_table.fkey_index(rel.fkey)[index], dtype=np.int64)
                blocks.append(Block(edge_list=np.stack([index, dst_index],axis=1), rel=rel))
                sampled_tables.append(dst_table[dst_index])
                index = dst_index
            elif rel.pkey_table is src_table:
                """
                1 - n
                For each src entry, sample all dst entries. O(k*n)
                """
                edges = np.empty((0, 2), dtype=np.int64)
                for i in index:
                    cur = np.where(dst_table.fkey_index(rel.fkey) == i)[0]
                    cur = np.stack([cur, np.full_like(cur, i, dtype=np.int64)], axis=1)
                    edges = np.concatenate((edges, cur), axis=0)
                blocks.append(Block(edge_list=edges, rel=rel))
                sampled_tables.append(dst_table[edges[:, 0]])
                index = edges[:, 0]
            else:
                raise ValueError("Invalid: Src_table: {}, Dst_table: {}, Rel: {}".format(src_table.table_name,
                                                                                        dst_table.table_name,
                                                                                        rel))

        return RelationFrame(tables=sampled_tables,
                             relation=self.rf.relations,
                             meta_graph=self.rf.meta_graph), blocks
    
        

