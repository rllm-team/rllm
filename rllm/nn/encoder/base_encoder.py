from abc import ABC
from typing import Callable

import torch
from torch import Tensor
import torch.nn.functional as F


class BaseEncoder(torch.nn.Module, ABC):
    r"""Shared base class for stacked encoders.

    This class provides common utilities used by encoder implementations:
    layer-count validation, a shared convolution container, and activation
    function resolution.
    """

    def __init__(self, num_layers: int) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")

        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()

    @staticmethod
    def get_activation(activation: str) -> Callable[[Tensor], Tensor]:
        activation_name = activation.lower()
        activation_map = {
            "relu": F.relu,
            "gelu": F.gelu,
            "elu": F.elu,
            "leaky_relu": F.leaky_relu,
            "selu": F.selu,
            "tanh": torch.tanh,
            "sigmoid": torch.sigmoid,
            "none": lambda x: x,
        }

        if activation_name not in activation_map:
            raise ValueError(
                f"Unsupported activation: {activation}. Supported: "
                "{'relu', 'gelu', 'elu', 'leaky_relu', 'selu', "
                "'tanh', 'sigmoid', 'none'}"
            )
        return activation_map[activation_name]
