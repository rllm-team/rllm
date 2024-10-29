import copy
from typing import Any


class BaseTransform:
    r"""An abstract base class for writing transforms."""
    def __call__(self, data: Any):
        # Shallow-copy the data so that we prevent in-place data modification.
        return self.forward(copy.copy(data))

    def forward(self, data: Any):
        pass

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'
