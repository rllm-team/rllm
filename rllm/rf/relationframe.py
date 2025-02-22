import sys
from typing import List
from dataclasses import dataclass
from functools import cached_property

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
        try:
            fkey_table_name = self.fkey_table.table_name
        except AttributeError:
            fkey_table_name = "fkey_table"
        try:
            pkey_table_name = self.pkey_table.table_name
        except AttributeError:
            pkey_table_name = "pkey_table"
        return f"{fkey_table_name}.{self.fkey} ----> {pkey_table_name}.{self.pkey}"


class RelationFrame:
    """
    RelationFrame is tables with relations
    """

    def __init__(
        self,
        tables: List[TableData],
        **kwargs,
    ):
        self.tables = tables

        if "relation" in kwargs:
            self._relations = kwargs["relation"]
        else:
            self._relations = self._infer_relation()

        if "meta_graph" in kwargs:
            self._meta_g = kwargs["meta_graph"]
        else:
            self._meta_g = self._construct_meta_graph()

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

    # funcs
    def _infer_relation(self):
        r"""
        Infer the relation between tables.
        """
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
        G = nx.DiGraph()
        for rel in self.relations:
            G.add_edge(rel.fkey_table, rel.pkey_table, relation=rel)
        return G
