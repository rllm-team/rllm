from abc import ABC, abstractmethod
import copy

from torch import Tensor

from rllm.data.graph_data import GraphData, HeteroGraphData


class NodeTransform(ABC):
    r"""An abstract base class for transforming nodes in graph data.
    It provides a common interface for all node transformation
    operations. It ensures that the data is shallow-copied to prevent
    in-place modifications.
    """

    def __call__(self, data):
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
    def forward(self, x):
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class EdgeTransform(ABC):
    r"""An abstract base class for transforming edges in graph data.
    It provides a common interface for all edge transformation
    operations. It ensures that the data is shallow-copied to prevent
    in-place modifications.
    """

    def __call__(self, data):
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
    def forward(self, adj):
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
