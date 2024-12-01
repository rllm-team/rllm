from typing import Union

from rllm.data.graph_data import GraphData, HeteroGraphData
from rllm.transforms.utils import BaseTransform
from rllm.transforms.utils.functional import svd_feature_reduction


class SVDFeatureReduction(BaseTransform):
    r"""Dimensionality reduction of node features via Singular Value
    Decomposition (SVD).

    Args:
        out_dim (int): The dimensionlity of node features after
            reduction.
    """

    def __init__(
        self,
        out_dim: int,
    ):
        self.out_dim = out_dim

    def forward(self, data: Union[GraphData, HeteroGraphData]):
        if isinstance(data, GraphData):
            assert data.x is not None
            data.x = svd_feature_reduction(data.x, self.out_dim)
        elif isinstance(data, HeteroGraphData):
            for store in data.node_stores:
                if "x" in store:
                    store.x = svd_feature_reduction(store.x, self.out_dim)

        return data
