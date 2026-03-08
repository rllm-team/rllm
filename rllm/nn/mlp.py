#  Copyright (c) Prior Labs GmbH 2025.

from __future__ import annotations

from functools import partial

import torch
from torch.utils.checkpoint import checkpoint


class MLP(torch.nn.Module):
    """Two-layer feed-forward block used in transformer-style models.

    This class applies `Linear -> Activation -> Linear` on the last dimension of
    the input tensor. It also supports optional residual addition and checkpointing.

    Args:
        dim: Input and output feature dimension.
        hidden_dim: Hidden layer feature dimension.
        activation: Activation name. Supported values: ``"gelu"`` and ``"relu"``.
        device: Device used to initialize model parameters.
        dtype: Data type used to initialize model parameters.
        initialize_output_to_zero:
            If ``True``, initialize the second linear layer weights to zero.
        recompute:
            If ``True``, enable activation checkpointing for lower memory usage.

    Returns:
        An initialized ``MLP`` module instance.

    Example:
        >>> mlp = MLP(
        ...     dim=128,
        ...     hidden_dim=256,
        ...     activation="gelu",
        ...     device=torch.device("cpu"),
        ...     dtype=torch.float32,
        ... )
        >>> x = torch.randn(4, 16, 128)
        >>> y = mlp(x)
        >>> y.shape
        torch.Size([4, 16, 128])
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        activation: str,
        *,
        device: torch.device | None,
        dtype: torch.dtype | None,
        initialize_output_to_zero: bool = False,
        recompute: bool = False,
    ):
        super().__init__()
        self.linear1 = torch.nn.Linear(
            dim,
            hidden_dim,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.linear2 = torch.nn.Linear(
            hidden_dim,
            dim,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.activation = activation.lower()
        if initialize_output_to_zero:
            torch.nn.init.zeros_(self.linear2.weight)
        if recompute:
            self.forward = partial(checkpoint, self.forward, use_reentrant=False)  # type: ignore

    def forward(
        self,
        x: torch.Tensor,
        *,
        add_input: bool = False,
        allow_inplace: bool = False,
    ) -> torch.Tensor:
        """Run the MLP transform on input features.

        Args:
            x:
                Input tensor with shape ``(..., dim)``.
            add_input:
                If ``True``, add residual connection (input + mlp(input)).
            allow_inplace:
                If ``True`` and ``add_input=True``, allow in-place residual add.

        Returns:
            Output tensor with the same shape as ``x``.

        Example:
            >>> mlp = MLP(
            ...     dim=32,
            ...     hidden_dim=64,
            ...     activation="relu",
            ...     device=torch.device("cpu"),
            ...     dtype=torch.float32,
            ... )
            >>> x = torch.randn(2, 5, 32)
            >>> y = mlp(x, add_input=True)
            >>> y.shape
            torch.Size([2, 5, 32])
        """
        input_shape = x.shape
        x = x.reshape(-1, x.size(-1))
        residual = x

        out = self.linear1(x)
        if self.activation == "gelu":
            out = torch.nn.functional.gelu(out)
        elif self.activation == "relu":
            out = torch.nn.functional.relu(out)
        else:
            raise NotImplementedError(
                f"Activation function '{self.activation}' is not implemented.",
            )
        out = self.linear2(out)

        if add_input:
            out = residual.add_(out) if allow_inplace else residual + out

        return out.reshape(input_shape)
