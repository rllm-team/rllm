from typing import Union

from rllm.data import GraphData, HeteroGraphData
from rllm.transforms.graph_transforms import BaseTransform
from rllm.transforms.utils.functional import normalize_features


class NormalizeFeatures(BaseTransform):
    r"""Row-normalizes the node features.

    .. math::

        \vec{x} = \frac{\vec{x}}{||\vec{x}||_p}

    Args:
        norm (str): The norm to use to normalize each non zero sample,
            *e.g.*, `l1`, `l2`. (default: `l2`)
    """

    def __init__(self, norm: str = "l2"):
        self.norm = norm

    def forward(self, data: Union[GraphData, HeteroGraphData]):
        for store in data.stores:
            if "x" in store:
                store.x = normalize_features(store.x, self.norm)
        return data
