import sys
from typing import List, Union, Tuple, Iterable
from dataclasses import dataclass
from functools import cached_property, wraps

import torch
from torch import Tensor
import pandas as pd
import numpy as np
import networkx as nx

sys.path.append("./")
sys.path.append("../")
from rllm.data.table_data import TableData


@dataclass
class Relation:
    """
    Relation shows the relationship between two tables.

    fkey_table.fkey ----> pkey_table.pkey
    """

    fkey_table: TableData
    fkey: str
    pkey_table: TableData
    pkey: str

    def __post_init__(self):
        assert self.fkey in self.fkey_table.cols, "Foreign key should be in the table."
        assert self.pkey in self.pkey_table.cols, "Primary key should be in the table."

    def __repr__(self):
        return f"{self.fkey_table.table_name}.{self.fkey} ----> {self.pkey_table.table_name}.{self.pkey}"

    def to_name(self):
        return NameRelation(self.fkey_table.table_name, self.fkey, self.pkey_table.table_name, self.pkey)


@dataclass
class NameRelation:
    """
    NameRelation shows the relationship between two tables.

    fkey_table.fkey ----> pkey_table.pkey
    """

    fkey_table: str
    fkey: str
    pkey_table: str
    pkey: str

    def __repr__(self):
        return f"{self.fkey_table}.{self.fkey} ----> {self.pkey_table}.{self.pkey}"


@dataclass(init=False)
class Block:
    # TODO: Arbitrary edge attributes
    """
    Edge direction: fkey_table.fkey ----> pkey_table.pkey
    edge_list: (src, dst), i.e., (fkey, pkey)
    It's worth noting that the direction of the edges here corresponds to the rel,
    while in the sampler, it may be reversed (for sample path is topological traveral).

    Args:
        edge_list (Union[Tuple[np.ndarray, np.ndarray], List[np.ndarray]]): A list of edges.
        nrel (Relation): The relation of the block.
        src_nodes_ (Union[np.ndarray, pd.Index]): The source nodes.
        dst_nodes_ (Union[np.ndarray, pd.Index]): The destination nodes.
    """
    edge_list: Union[Tuple[np.ndarray, np.ndarray], List[np.ndarray]]
    # edge_val = Tensor
    nrel: NameRelation
    src_nodes_: Union[np.ndarray, pd.Index]
    dst_nodes_: Union[np.ndarray, pd.Index]

    def __init__(
        self,
        edge_list: Union[Tuple[np.ndarray, np.ndarray], List[np.ndarray]],
        nrel: NameRelation,
        src_nodes_: Union[np.ndarray, pd.Index],
        dst_nodes_: Union[np.ndarray, pd.Index],
    ):
        self.edge_list = edge_list
        self.nrel = nrel
        self.src_nodes_ = np.unique(src_nodes_)
        self.dst_nodes_ = np.unique(dst_nodes_)

    @property
    def src_nodes(self):
        return self.src_nodes_

    @property
    def dst_nodes(self):
        return self.dst_nodes_

    def __repr__(self):
        n_src_nodes = len(self.src_nodes)
        n_dst_nodes = len(self.dst_nodes)
        repr = f"""Block:[
        nrel: {self.nrel}
        num_edges: {len(self.edge_list[0])}
        num_src_nodes: {n_src_nodes}
        num_dst_nodes: {n_dst_nodes}]"""
        return repr


class RelationFrame:
    """
    RelationFrame is tables with relations

    Args:
        tables (List[TableData]): A list of tables.
        relations (Optional[List[Relation]]): A list of Relations.
        meta_graph (Optional[nx.DiGraph]): A directed graph to represent the relations.

        sampled (bool): Whether the relation frame is sampled.
        _blocks (Optional[List[Block]]): A list of blocks to represent the sampling edges.
        sampling_seeds (Optional[Interable]): Sampled batch index.
        target_table_name (Optional[TableData]): The target table for sampling.
    """
    init_attrs = ["relations", "meta_graph"]
    init_mapping = {"relations": "_infer_relation", "meta_graph": "_construct_meta_graph"}

    sampled: bool = False
    sampled_attrs = ["_blocks", "sampling_seeds", "target_table_name"]

    def __init__(
        self,
        tables: List[TableData],
        **kwargs,
    ):
        self.tables = tables

        self._init(**kwargs)
        if 'sampled' in kwargs and kwargs['sampled']:
            self.sampled = True
            self._sampled_init(**kwargs)

    def _init(self, **kwargs):
        for k in self.init_attrs:
            if k in kwargs:
                setattr(self, k, kwargs.pop(k))
            elif k in self.init_mapping:
                setattr(self, k, getattr(self, self.init_mapping[k])())

    def __setattr__(self, name, value):
        allowed_attrs = ['tables', 'sampled'] + self.init_attrs + self.sampled_attrs
        if name in allowed_attrs:
            super().__setattr__(name, value)
        else:
            raise AttributeError(f"Cannot set invalid attribute {name}.")

    @cached_property
    def undirected_meta_graph(self):
        # as_view=True to keep Relation and TableData.
        return self.meta_graph.to_undirected(as_view=True)

    # funcs
    def reset_table_index(self):
        r"""
        Reset the index of tables. This func will overwrite the original index.
        """
        for table in self.tables:
            if isinstance(table.df.index, pd.RangeIndex):
                continue
            o_index_name = table.index_col
            table.df.reset_index(drop=True, inplace=True)
            table.df.index.name = o_index_name

    def _infer_relation(self):
        r"""
        Infer the relation between tables.
        """
        if len(self.tables) == 1:
            return None

        relations = []
        for fkey_table in self.tables:
            for col in fkey_table.cols:
                if col == fkey_table.index_col:
                    continue
                for pkey_table in self.tables:
                    if pkey_table != fkey_table and col == pkey_table.index_col:
                        relations.append(Relation(fkey_table, col, pkey_table, col))
        return relations

    def _construct_meta_graph(self):
        r"""
        Construct a meta graph from the relations.
        """
        G = nx.MultiDiGraph()
        for rel in self.relations:
            # G.add_edge((rel.fkey_table, rel.fkey), (rel.pkey_table, rel.pkey), relation=rel)
            G.add_edge(rel.fkey_table, rel.pkey_table, relation=rel)
        return G

    # extra funcs
    def validate_rels(self) -> bool:
        r"""
        Validate the relations in the relation frame.
        """
        for rel in self.relations:
            assert rel.fkey_table in self.tables, f"{rel.fkey_table} should be in tables."
            assert rel.pkey_table in self.tables, f"{rel.pkey_table} should be in tables."
            assert rel.fkey in rel.fkey_table.cols, f"{rel.fkey} should be in {rel.fkey_table}."
            assert rel.pkey in rel.pkey_table.cols, f"{rel.pkey} should be in {rel.pkey_table}."
        return True

    def unify_f_pkey_dtype(self) -> None:
        r"""
        Unify the data type of foreign key and primary key.
        """
        for rel in self.relations:
            fkey_dtype = rel.fkey_table.df[rel.fkey].dtype
            pkey_dtype = rel.pkey_table.df.index.dtype
            if fkey_dtype != pkey_dtype:
                try:
                    rel.fkey_table.df[rel.fkey] = rel.fkey_table.df[rel.fkey].astype(pkey_dtype)
                except ValueError:
                    raise ValueError(f"Failed to convert {rel.fkey} to {pkey_dtype}.")

    def validate_rels_deep(self):
        r"""
        Validate the entry links of relations.
        This func is time consuming for large tables.
        """
        # TODO
        return True

    # after sampled, utils
    def after_sampled(func):
        @wraps(func)
        def warpper(self, *args, **kwargs):
            if not self.sampled:
                raise ValueError("Only accessible after sampled.")
            return func(self, *args, **kwargs)
        return warpper

    @after_sampled
    def _sampled_init(self, **kwargs):
        for k in self.sampled_attrs:
            if k in kwargs:
                setattr(self, k, kwargs.pop(k))

    @property
    @after_sampled
    def nrelations(self):
        if isinstance(self._relations[0], NameRelation):
            return self._relations
        else:
            raise TypeError("Relation in rf is Relation, call relations instead.")

    @cached_property
    @after_sampled
    def blocks(self):
        if hasattr(self, '_blocks'):
            return self._blocks
        else:
            raise AttributeError("Blocks is not set.")

    @cached_property
    @after_sampled
    def dict_blocks(self):
        return {f"{block.nrel}": block for block in self.blocks}

    @cached_property
    @after_sampled
    def target_table(self):
        for table in self.tables:
            if table.table_name == self.target_table_name:
                return table

    @after_sampled
    def oind2ind(self, index_l: Iterable, oind: List[int] = None) -> List[int]:
        r"""Transform the original index to the index in the sampling seeds."""
        oind = self.target_table.oind if oind is None else oind
        ind = []
        for i in index_l:
            ind.append(oind.index(i))
        return ind

    @cached_property
    @after_sampled
    def target_index(self) -> Tensor:
        return torch.tensor(self.oind2ind(self.sampling_seeds), dtype=torch.long)

    @cached_property
    @after_sampled
    def y(self):
        return self.target_table.y[self.target_index]
