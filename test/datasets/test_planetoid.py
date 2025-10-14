from typing import Dict
from torch import Tensor

from rllm.datasets.planetoid import PlanetoidDataset
from rllm.data.graph_data import GraphData


def test_process():
    dataset_names = ["cora", "citeseer", "pubmed"]
    for name in dataset_names:
        dataset = PlanetoidDataset("./data", file_name=name, force_reload=True)
        data = dataset[0]
        assert isinstance(data, GraphData)

        data.cpu()
        data_copy = data.clone()
        assert id(data) != id(data_copy)

        adj = data.adj
        x = data.x
        num_classes = data.num_classes
        num_nodes = data.num_nodes
        data_dict = data.to_dict()
        assert isinstance(adj, Tensor)
        assert isinstance(x, Tensor)
        assert isinstance(num_classes, int)
        assert isinstance(num_nodes, int)
        assert isinstance(data_dict, Dict)
