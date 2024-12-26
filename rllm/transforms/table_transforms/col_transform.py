from abc import ABC, abstractmethod
import copy


class ColTransform(ABC):
    r"""An abstract base class for writing transforms."""

    def __call__(self, data):
        # Shallow-copy the data so that we prevent in-place data modification.
        return self.forward(copy.copy(data))

    @abstractmethod
    def forward(self, data):
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
