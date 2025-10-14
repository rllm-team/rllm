import pandas as pd
import numpy as np
import torch

from rllm.types import ColType
from rllm.data import HeteroGraphData, TableData


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
