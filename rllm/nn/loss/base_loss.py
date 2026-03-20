from __future__ import annotations

import torch
import torch.nn


class BaseLoss(torch.nn.Module):
    r"""
    Minimal root class for all custom loss functions in this repository.

    Rationale:
    - It is a single, consistent parent type so that higher-level code
      (trainer loops, registries, logging utilities) can treat every
      project-specific loss in a uniform way.
    - Concrete subclasses must implement `forward(...)` and return a
      scalar tensor.

    This class itself does not impose any particular training logic.
    It only standardizes the interface.

    Args:
      No additional constructor arguments.

    Example:
      >>> import torch
      >>> class DummyLoss(BaseLoss):
      ...     def forward(self, x):
      ...         return x.mean()
      >>> loss_fn = DummyLoss()
      >>> loss_fn(torch.tensor([1.0, 2.0, 3.0]))
      tensor(2.)
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Compute the loss value.

        Args:
          *args: Positional inputs defined by subclasses.
          **kwargs: Keyword inputs defined by subclasses.

        Returns:
          torch.Tensor: A scalar loss tensor.

        Example:
          >>> class DummyLoss(BaseLoss):
          ...     def forward(self, x):
          ...         return x.sum()
          >>> DummyLoss()(torch.tensor([1.0, 2.0]))
          tensor(3.)
        """

        raise NotImplementedError(f"{self.__class__.__name__} must implement forward()")
