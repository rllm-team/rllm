import sys
from typing import List, Union, Tuple
from dataclasses import dataclass
from functools import cached_property

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
        rel (Relation): The relation of the block.
        src_nodes_ (Union[np.ndarray, pd.Index]): The source nodes.
        dst_nodes_ (Union[np.ndarray, pd.Index]): The destination nodes.
    """
    edge_list: Union[Tuple[np.ndarray, np.ndarray], List[np.ndarray]]
    # edge_val = Tensor
    rel: Relation
    src_nodes_: Union[np.ndarray, pd.Index]
    dst_nodes_: Union[np.ndarray, pd.Index]

    def __init__(
        self,
        edge_list: Union[Tuple[np.ndarray, np.ndarray], List[np.ndarray]],
        rel: Relation,
        src_nodes_: Union[np.ndarray, pd.Index],
        dst_nodes_: Union[np.ndarray, pd.Index],
    ):
        self.edge_list = edge_list
        self.rel = rel
        self.src_nodes_ = np.unique(src_nodes_)
        self.dst_nodes_ = np.unique(dst_nodes_)
        pass

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
        rel: {self.rel}
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
        _blocks (Optional[List[Block]]): A list of blocks to represent the sampling edges.
        **kwargs: Additional keyword arguments.
    """

    def __init__(
        self,
        tables: List[TableData],
        **kwargs,
    ):
        self.tables = tables

        if "relations" in kwargs:
            self._relations = kwargs["relations"]
        else:
            self._relations = self._infer_relation()

        if "meta_graph" in kwargs:
            self._meta_g = kwargs["meta_graph"]
        else:
            self._meta_g = self._construct_meta_graph()

        if "_blocks" in kwargs and isinstance(kwargs["_blocks"][0], Block):
            self._blocks = kwargs["_blocks"]
            self.reset_block_relation()

    def __setattr__(self, name, value):
        allowed_attrs = {"tables", "_relations", "_meta_g", "_blocks"}
        if name in allowed_attrs:
            super().__setattr__(name, value)
        else:
            raise AttributeError(f"Cannot set invalid attribute {name}.")

    @property
    def relations(self):
        return self._relations

    @property
    def meta_graph(self):
        return self._meta_g

    @cached_property
    def undirected_meta_graph(self):
        # as_view=True to keep Relation and TableData.
        return self._meta_g.to_undirected(as_view=True)

    @cached_property
    def blocks(self):
        if hasattr(self, '_blocks'):
            return self._blocks
        else:
            raise AttributeError("Blocks is not set.")

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

    def reset_block_relation(self):
        r"""
        Called after sampling, reset the relations in blocks.
        """
        for block in self._blocks:
            f_t = block.rel.fkey_table.table_name
            p_t = block.rel.pkey_table.table_name
            for rel in self.relations:
                if f_t == rel.fkey_table.table_name and p_t == rel.pkey_table.table_name:
                    block.rel = rel
                    break

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
        pass
        return True
