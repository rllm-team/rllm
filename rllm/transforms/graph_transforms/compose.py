from typing import Callable, List, Union

from rllm.data.graph_data import GraphData, HeteroGraphData
from rllm.transforms.graph_transforms import NETransform


class Compose(NETransform):
    r"""Composes several NETransforms together into a single Transform.
    This class allows for the sequential application of a list of NETransforms
    to graph data. It is particularly useful in scenarios where multiple
    preprocessing steps need to be applied to graph data in a specific order.

    Args:
        transforms (List[Callable]): List of transforms to compose.
    """

    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def forward(self, data: Union[GraphData, HeteroGraphData]):
        for transform in self.transforms:
            if isinstance(data, (list, tuple)):
                data = [transform(d) for d in data]
            else:
                data = transform(data)
        return data
