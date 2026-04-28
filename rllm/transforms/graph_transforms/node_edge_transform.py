import copy
from abc import ABC, abstractmethod
from typing import Union

from torch import Tensor

from rllm.data.graph_data import GraphData, HeteroGraphData


class NodeTransform(ABC):
    r"""Base class for node-wise transformations on graph data.

    The transform is applied to ``x`` for homogeneous graphs, each valid
    ``store.x`` for heterogeneous graphs, or directly to a square
    :class:`torch.Tensor`.

    Shape:
        - ``GraphData``: ``data.x`` can be any node feature shape accepted by
          subclasses.
        - ``HeteroGraphData``: ``store.x`` can be any node feature shape
          accepted by subclasses.
        - :class:`torch.Tensor`: input must be a square matrix with shape
          ``[N, N]``.

    Examples::

        class NormalizeNodeX(NodeTransform):
            def forward(self, x):
                return x / (x.norm(dim=-1, keepdim=True) + 1e-12)

        transform = NormalizeNodeX()
        out = transform(data)
    """

    def __call__(self, data: Union[GraphData, HeteroGraphData, Tensor]):
        # Shallow-copy the data so that we prevent in-place data modification.
        data = copy.copy(data)
        if isinstance(data, GraphData):
            if getattr(data, "x", None) is not None:
                data.x = self.forward(data.x)
        elif isinstance(data, HeteroGraphData):
            for store in data.node_stores:
                if "x" not in store or not store.is_bipartite():
                    continue
                store.x = self.forward(store.x)
        elif isinstance(data, Tensor):
            assert data.size(0) == data.size(1)
            data = self.forward(data)

        return data

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class EdgeTransform(ABC):
    r"""Base class for edge-wise transformations on graph data.

    The transform is applied to ``adj`` for homogeneous graphs, each valid
    ``store.adj`` for heterogeneous graphs, or directly to a tensor input.

    Shape:
        - ``GraphData``: ``data.adj`` can be dense or sparse and should follow
          the adjacency format expected by subclasses.
        - ``HeteroGraphData``: ``store.adj`` can be dense or sparse and should
          follow the adjacency format expected by subclasses.
        - :class:`torch.Tensor`: if dense, input must be a square matrix with
          shape ``[N, N]``; sparse tensors are forwarded as-is.

    Examples::

        class KeepSelfLoops(EdgeTransform):
            def forward(self, adj):
                return adj

        transform = KeepSelfLoops()
        out = transform(data)
    """

    def __call__(self, data: Union[GraphData, HeteroGraphData, Tensor]):
        # Shallow-copy the data so that we prevent in-place data modification.
        data = copy.copy(data)

        if isinstance(data, GraphData):
            if getattr(data, "adj", None) is not None:
                data.adj = self.forward(data.adj)
        elif isinstance(data, HeteroGraphData):
            for store in data.edge_stores:
                if "adj" not in store or not store.is_bipartite():
                    continue
                store.adj = self.forward(store.adj)
        elif isinstance(data, Tensor):
            if not data.is_sparse:
                assert data.size(0) == data.size(1)
            data = self.forward(data)

        return data

    @abstractmethod
    def forward(self, adj: Tensor) -> Tensor:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
