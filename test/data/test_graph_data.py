import pandas as pd
import numpy as np
import torch

from rllm.types import ColType
from rllm.data import EdgeStorage, HeteroGraphData, TableData


def test_heterographdata():
    hgraph = HeteroGraphData()
    hgraph[("node0", "aaaa", "node1")].edge_index = torch.tensor([[0, 1], [1, 0]])
    # hgraph['node0'].x = torch.tensor([1, 2])
    df = pd.DataFrame(
        {
            "f1": np.ones(2),
            "f2": np.zeros(2),
        }
    )
    table = TableData(
        df,
        col_types={
            "f1": ColType.NUMERICAL,
            "f2": ColType.CATEGORICAL,
        },
        lazy_feature=True,
    )
    table.lazy_materialize(keep_df=False)
    hgraph["node0"].x = table
    hgraph["node1"].x = torch.tensor([1.1, 2.1])
    assert hgraph.validate()

    hgraph["node0"].x = None
    assert len(hgraph.collect_attr("x", exlude_None=False)) == 2
    assert len(hgraph.collect_attr("x", exlude_None=True)) == 1

    import copy

    hgraph2 = copy.copy(hgraph)
    assert id(hgraph2["node1"].x) == id(hgraph["node1"].x)
    hgraph2["node1"].x = torch.tensor([1.2, 2.2])
    assert id(hgraph2["node1"].x) != id(hgraph["node1"].x)


def test_heterographdata_to_csc_dict_uses_dst_num_nodes():
    hgraph = HeteroGraphData()
    hgraph["src"].num_nodes = 2
    hgraph["dst"].num_nodes = 4
    hgraph[("src", "to", "dst")].edge_index = torch.tensor(
        [[0, 1], [0, 3]], dtype=torch.long
    )

    col_ptr_d, row_d, perm_d = hgraph.to_csc_dict()

    edge_type = ("src", "to", "dst")
    assert torch.equal(col_ptr_d[edge_type], torch.tensor([0, 1, 1, 1, 2]))
    assert torch.equal(row_d[edge_type], torch.tensor([0, 1]))
    assert perm_d[edge_type] is not None


def test_edge_storage_is_edge_attr_accepts_tensor_weight():
    storage = EdgeStorage()
    storage.edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    storage.weight = torch.tensor([0.1, 0.2])

    assert storage.is_edge_attr("weight")
