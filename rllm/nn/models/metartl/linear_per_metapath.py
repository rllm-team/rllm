import math

import torch
from torch import Tensor
from torch import nn


def _grouped_xavier_uniform_(tensor: Tensor, gain: float = 1.0) -> Tensor:
    r"""Xavier uniform initialization that operates on the last two dims.

    Unlike :func:`torch.nn.init.xavier_uniform_`, this supports parameters with
    a leading group dimension (e.g. :obj:`[num_metapaths, cin, cout]`) by using
    the trailing two dimensions to compute the fan-in/fan-out.

    Args:
        tensor (Tensor): Parameter to be initialized.
        gain (float): Gain factor. (default: :obj:`1.0`)

    Returns:
        Tensor: The initialized parameter.
    """
    fan_in, fan_out = tensor.size()[-2:]
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    a = math.sqrt(3.0) * std
    with torch.no_grad():
        return tensor.uniform_(-a, a)


class LinearPerMetapath(nn.Module):
    r"""Per-meta-path linear projection used in the semantic feature
    transformation of MetaRTL.

    Applies an independent linear layer to each of the :obj:`num_metapaths`
    meta-path slices of the input, i.e. the einsum :obj:`'bcm,cmn->bcn'` with a
    per-meta-path weight and bias.

    Args:
        in_dim (int): Input feature dimensionality.
        out_dim (int): Output feature dimensionality.
        num_metapaths (int): Number of meta-paths (the second input dimension).

    Example:
        >>> import torch
        >>> layer = LinearPerMetapath(16, 32, num_metapaths=5)
        >>> layer(torch.randn(8, 5, 16)).shape
        torch.Size([8, 5, 32])
    """

    def __init__(self, in_dim: int, out_dim: int, num_metapaths: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_metapaths = num_metapaths

        self.weight = nn.Parameter(torch.randn(num_metapaths, in_dim, out_dim))
        self.bias = nn.Parameter(torch.zeros(num_metapaths, out_dim))

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        gain = torch.nn.init.calculate_gain("relu")
        _grouped_xavier_uniform_(self.weight, gain=gain)
        torch.nn.init.zeros_(self.bias)

    def forward(self, x: Tensor) -> Tensor:
        return torch.einsum("bcm,cmn->bcn", x, self.weight) + self.bias.unsqueeze(0)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.in_dim}, {self.out_dim}, "
            f"num_metapaths={self.num_metapaths})"
        )
