from abc import ABC
from typing import Callable, Dict, Optional, Type, Union

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

    @staticmethod
    def build_layer_dims(
        in_dim: int,
        out_dim: int,
        hidden_dim: Optional[int],
        hidden_default_dim: int,
        num_layers: int,
    ) -> list[int]:
        hidden_dim = hidden_default_dim if hidden_dim is None else hidden_dim
        return [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]

    @staticmethod
    def resolve_norm_layer(norm_layer: Optional[str]) -> Type[torch.nn.Module]:
        norm_layer_name = "none" if norm_layer is None else norm_layer.lower()
        if norm_layer_name == "layernorm":
            return torch.nn.LayerNorm
        if norm_layer_name in {"batchnorm", "batchnorm1d"}:
            return torch.nn.BatchNorm1d
        if norm_layer_name == "none":
            return torch.nn.Identity
        raise ValueError(
            "Unsupported norm_layer: "
            f"{norm_layer}. Supported: {'layernorm', 'batchnorm1d', 'none'}"
        )

    @staticmethod
    def validate_feature_last_dim(
        x: Union[Tensor, Dict[object, Tensor]],
        expected_dim: int,
        allow_dict: bool = True,
    ) -> None:
        if isinstance(x, dict):
            if allow_dict:
                return
            for feat in x.values():
                if feat.size(-1) != expected_dim:
                    raise ValueError(
                        "Feature dimension mismatch: "
                        f"expected {expected_dim}, got {feat.size(-1)}."
                    )
            return

        if x.size(-1) != expected_dim:
            raise ValueError(
                "Feature dimension mismatch: "
                f"expected {expected_dim}, got {x.size(-1)}."
            )
