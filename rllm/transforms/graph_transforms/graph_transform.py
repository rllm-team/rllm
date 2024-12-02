from __future__ import annotations
from abc import ABC
from typing import List, Callable, Union

from torch.nn import Module

from rllm.data.graph_data import GraphData, HeteroGraphData


class GraphTransform(Module, ABC):
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
