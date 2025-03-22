from typing import Union

from torch import Tensor

from rllm.data import GraphData, HeteroGraphData
from rllm.transforms.graph_transforms import NodeTransform
from rllm.transforms.graph_transforms.functional import normalize_features


class NormalizeFeatures(NodeTransform):
    r"""Row-normalizes the node features.

    .. math::

        \vec{x} = \frac{\vec{x}}{||\vec{x}||_p}

    Args:
        norm (str): The norm to use to normalize each non zero sample,
            *e.g.*, `l1`, `l2`. (default: `l2`)
    """

    def __init__(self, norm: str = "l2"):
        self.norm = norm

    def forward(self, data: Union[Tensor, GraphData, HeteroGraphData]):
        if isinstance(data, Tensor):
            return normalize_features(data)

        for store in data.stores:
            if "x" in store:
                store.x = normalize_features(store.x, self.norm)
        return data
