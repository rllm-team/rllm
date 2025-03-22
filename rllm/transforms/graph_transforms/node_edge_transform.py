from abc import ABC, abstractmethod
import copy


class NodeTransform(ABC):
    r"""An abstract base class for transforming nodes in graph data.
    It provides a common interface for all node transformation
    operations. It ensures that the data is shallow-copied to prevent
    in-place modifications.
    """

    def __call__(self, data):
        # Shallow-copy the data so that we prevent in-place data modification.
        return self.forward(copy.copy(data))

    @abstractmethod
    def forward(self, data):
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
        return self.forward(copy.copy(data))

    @abstractmethod
    def forward(self, data):
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
