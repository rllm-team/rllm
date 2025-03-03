from typing import Iterable
from abc import ABC, abstractmethod

from rllm.data import RelationFrame, TableData


class BaseSampler(ABC):
    r"""Base class for samplers.

    Args:
        rf: RelationFrame
        target_table: TableData
    """

    rf: RelationFrame
    target_table: TableData

    @abstractmethod
    def sample(self, index: Iterable) -> RelationFrame:
        r"""Samples a batch of data."""
        raise NotImplementedError
