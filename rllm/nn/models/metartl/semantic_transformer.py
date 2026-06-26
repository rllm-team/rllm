from typing import Optional

import torch
from torch import Tensor
from torch import nn


class SemanticTransformer(nn.Module):
    r"""Transformer-based semantic fusion over meta-paths.

    Treats the :obj:`M` meta-path embeddings of each node as a sequence and
    mixes information across them with a standard
    :class:`torch.nn.MultiheadAttention` self-attention block, scaled by a
    learnable residual gate :obj:`gamma` (initialized to zero so the module
    starts as identity).

    Args:
        n_channels (int): Input/output feature dimensionality.
        num_heads (int): Number of attention heads. (default: :obj:`1`)
        att_drop (float): Dropout on the attention weights. (default: :obj:`0.0`)

    Example:
        >>> import torch
        >>> attn = SemanticTransformer(32, num_heads=4)
        >>> attn(torch.randn(8, 5, 32)).shape
        torch.Size([8, 5, 32])
    """

    def __init__(
        self,
        n_channels: int,
        num_heads: int = 1,
        att_drop: float = 0.0,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.num_heads = num_heads

        self.attn = nn.MultiheadAttention(
            embed_dim=n_channels,
            num_heads=num_heads,
            dropout=att_drop,
            batch_first=True,
        )
        self.gamma = nn.Parameter(torch.tensor([0.0]))

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.attn._reset_parameters()  # type: ignore[reportPrivateUsage]
        torch.nn.init.zeros_(self.gamma)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        r"""Apply semantic self-attention across meta-paths.

        Args:
            x (Tensor): Input of shape :obj:`[B, M, C]` where :obj:`M` is the
                number of meta-paths.
            mask (Optional[Tensor]): Optional key padding mask of shape
                :obj:`[B, M]` (:obj:`True` marks positions to ignore).

        Returns:
            Tensor: Output of shape :obj:`[B, M, C]` (with residual connection).
        """
        attn_out, _ = self.attn(x, x, x, key_padding_mask=mask, need_weights=False)
        return self.gamma * attn_out + x

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.n_channels}, "
            f"num_heads={self.num_heads})"
        )
