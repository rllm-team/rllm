import math

import torch
from torch import Tensor
from torch.nn import Module


class CyclicEncoder(Module):
    r"""Cyclic encoding from paper 
    `"Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains"
    <https://arxiv.org/abs/2006.10739>`_. 
    For input data containing values between 0 and 1,
    this function maps each value in the input using sine and cosine
    functions of different wavelengths to preserve the cyclical nature. This
    is particularly useful for encoding cyclical features like hours of a
    day, days of the week, etc. Given an input tensor of shape
    :obj:`(*, )`, this encoding expands it into an output tensor of shape
    :obj:`(*, out_size)`.

    Args:
        out_size (int): The output dimension size.

    Returns:
        This class does not return a tensor in ``__init__``.
        The ``forward`` method returns encoded tensor with shape
        ``input_tensor.shape + (out_size,)``.

    Example:
        >>> import torch
        >>> enc = CyclicEncoder(out_size=8)
        >>> x = torch.rand(2, 3)
        >>> enc(x).shape
        torch.Size([2, 3, 8])
    """
    def __init__(self, out_size: int) -> None:
        super().__init__()
        if out_size % 2 != 0:
            raise ValueError(
                f"out_size should be divisible by 2 (got {out_size}).")
        self.out_size = out_size
        self.mult_term: Tensor
        self.register_buffer(
            "mult_term",
            torch.arange(1, self.out_size // 2 + 1),
        )

    def reset_parameters(self) -> None:
        r"""This is a placeholder function for compatibility."""
        pass

    def forward(self, input_tensor: Tensor) -> Tensor:
        assert torch.all((input_tensor >= 0) & (input_tensor <= 1))
        mult_tensor = input_tensor.unsqueeze(-1) * self.mult_term.reshape(
            (1, ) * input_tensor.ndim + (-1, ))
        return torch.cat([
            torch.sin(mult_tensor * math.pi),
            torch.cos(mult_tensor * 2 * math.pi)
        ], dim=-1)