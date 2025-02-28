from abc import ABC
from typing import Any


class BaseSampler(ABC):
    r"""
    BaseSampler is the base class for all samplers.
    """
    def sample(self, *args, **kwargs) -> Any:
        raise NotImplementedError
