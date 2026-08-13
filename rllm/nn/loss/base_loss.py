from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class BaseLoss(torch.nn.Module, ABC):
    r"""Minimal root class for all custom loss functions in this repository.

    Rationale:
    - Provides a single, consistent parent type so that higher-level code
      (trainer loops, registries, logging utilities) can treat every
      project-specific loss uniformly.
    - Concrete subclasses must implement ``forward(...)`` and return a
      scalar tensor.  Attempting to instantiate a subclass that does not
      override ``forward`` will raise ``TypeError`` at construction time
      (thanks to ``@abstractmethod``).

    Example:
        >>> import torch
        >>> class DummyLoss(BaseLoss):
        ...     def forward(self, x):
        ...         return x.mean()
        >>> loss_fn = DummyLoss()
        >>> loss_fn(torch.tensor([1.0, 2.0, 3.0]))
        tensor(2.)
    """

    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Compute the loss value.

        Args:
            *args: Positional inputs defined by subclasses.
            **kwargs: Keyword inputs defined by subclasses.

        Returns:
            torch.Tensor: A scalar loss tensor.
        """
