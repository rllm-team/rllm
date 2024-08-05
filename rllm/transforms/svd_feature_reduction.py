from typing import Union

from rllm.transforms.utils import svd_feature_reduction
from rllm.data.graph_data import GraphData, HeteroGraphData
from rllm.transforms.base_transform import BaseTransform


class SVDFeatureReduction(BaseTransform):
    r"""Dimensionality reduction of node features via Singular Value
    Decomposition (SVD).

    Args:
        out_channels (int): The dimensionlity of node features after
            reduction.
    """
    def __init__(self, out_channels: int):
        self.out_channels = out_channels

    def forward(self, data: Union[GraphData, HeteroGraphData]):
        if isinstance(data, GraphData):
            assert data.x is not None
            data.x = svd_feature_reduction(data.x, self.out_channels)
        elif isinstance(data, HeteroGraphData):
            for store in data.node_stores:
                if 'x' in store:
                    store.x = svd_feature_reduction(store.x, self.out_channels)

        return data
