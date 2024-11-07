from typing import Dict, List, Tuple

from rllm.datasets.dblp import DBLP
from rllm.data.graph_data import HeteroGraphData


def test_attribute():
    dataset = DBLP("./data", force_reload=True)
    data = dataset[0]
    assert isinstance(data, HeteroGraphData)

    data.cpu()
    data_copy = data.clone()
    assert id(data) != id(data_copy)

    adj_dict = data.adj_dict()
    edge_items = data.edge_items()
    node_items = data.node_items()
    metadata = data.metadata()
    x_dict = data.x_dict()
    data_dict = data.to_dict()

    assert isinstance(adj_dict, Dict)
    assert isinstance(edge_items, List)
    assert isinstance(node_items, List)
    assert isinstance(metadata, Tuple)
    assert isinstance(x_dict, Dict)
    assert isinstance(data_dict, Dict)

    edge_types = data.edge_types
    node_types = data.node_types
    num_node = data.num_nodes

    assert isinstance(edge_types, List)
    assert isinstance(node_types, List)
    assert isinstance(num_node, int)
