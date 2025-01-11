from __future__ import annotations
from abc import ABC
from typing import List, Callable, Union

import torch

from rllm.data.graph_data import GraphData, HeteroGraphData


class GraphTransform(torch.nn.Module, ABC):
    r"""The GraphTransform class is a base class for applying a series of
    transformations to graph data. It supports both homogeneous and
    heterogeneous graph data.

    Args:
        transforms (List[Callable]): A list of transformation functions to be
            applied to the graph data.
    """

    def __init__(
        self,
        transforms: List[Callable],
    ) -> None:
        super().__init__()
        self.data = None
        self.transforms = transforms

    def forward(
        self,
        data: Union[GraphData, HeteroGraphData],
    ):
        for transform in self.transforms:
            if isinstance(data, (list, tuple)):
                data = [transform(d) for d in data]
            else:
                data = transform(data)
        return data
