from abc import ABC, abstractmethod
import copy

from torch import Tensor

from rllm.data.graph_data import GraphData, HeteroGraphData


class NodeTransform(ABC):
    r"""Base class for node-level transforms.

    This class applies :meth:`forward` to node features while keeping the input
    container immutable via shallow copy.

    Args:
        None.

    Shape:
        - Input: :class:`~rllm.data.graph_data.GraphData`,
          :class:`~rllm.data.graph_data.HeteroGraphData`, or
          :class:`torch.Tensor` with shape ``[num_nodes, num_features]``.
        - Output: Same type as input. Feature shape is determined by the
          concrete transform.

    Examples:
        >>> transform = NormalizeFeatures("l2")
        >>> data = transform(data)
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
    r"""Base class for edge-level transforms.

    This class applies :meth:`forward` to adjacency matrices while keeping the
    input container immutable via shallow copy.

    Args:
        None.

    Shape:
        - Input: :class:`~rllm.data.graph_data.GraphData`,
          :class:`~rllm.data.graph_data.HeteroGraphData`, or square
          :class:`torch.Tensor` with shape ``[num_nodes, num_nodes]``.
        - Output: Same type as input. Adjacency shape remains
          ``[num_nodes, num_nodes]``.

    Examples:
        >>> transform = RemoveSelfLoops()
        >>> data = transform(data)
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
