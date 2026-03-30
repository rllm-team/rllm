from __future__ import annotations
from typing import Optional

import torch
from torch import Tensor


def _get_activation_fn(activation):
    if activation == "relu":
        return torch.nn.functional.relu
    elif activation == "gelu":
        return torch.nn.functional.gelu
    elif activation == "selu":
        return torch.nn.functional.selu
    elif activation == "leakyrelu":
        return torch.nn.functional.leaky_relu
    raise RuntimeError(
        "activation should be relu/gelu/selu/leakyrelu, not {}".format(activation)
    )


class TransTabConv(torch.nn.Module):
    r"""Single Transformer encoder layer for TransTab
    (`"TransTab" <https://arxiv.org/abs/2205.09328>`_).

    Combines multi-head self-attention with a gated feedforward network,
    residual connections, dropout, and optional LayerNorm.

    Args:
        conv_dim (int): Input/output embedding dimensionality.
        nhead (int): Number of self-attention heads.
        dim_feedforward (int): Feedforward inner dimension. Default: ``2048``.
        dropout (float): Dropout probability. Default: ``0.1``.
        activation (str or Callable): Feedforward activation; accepts
            ``"relu"``, ``"gelu"``, ``"selu"``, ``"leakyrelu"``, or a callable.
            Default: ``torch.nn.functional.relu``.
        layer_norm_eps (float): LayerNorm :math:`\varepsilon`. Default: ``1e-5``.
        batch_first (bool): Expect input as :math:`(N, S, H)` when ``True``,
            else :math:`(S, N, H)`. Default: ``True``.
        norm_first (bool): Apply LayerNorm before (pre-norm) rather than after
            (post-norm) each sub-layer. Default: ``False``.
        use_layer_norm (bool): Include LayerNorm in each sub-block. Default: ``True``.

    Shape:
        - Input: :math:`(N, S, H)` when ``batch_first=True``.
        - Output: :math:`(N, S, H)`.

    Examples::

        >>> conv = TransTabConv(conv_dim=32, nhead=4, dim_feedforward=64)
        >>> out = conv(torch.randn(8, 10, 32), src_key_padding_mask=torch.ones(8, 10))
        >>> out.shape
        torch.Size([8, 10, 32])
    """

    __constants__ = ["batch_first", "norm_first"]

    def __init__(
        self,
        conv_dim,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation=torch.nn.functional.relu,
        layer_norm_eps=1e-5,
        batch_first=True,
        norm_first=False,
        use_layer_norm=True,
    ) -> None:
        super().__init__()
        self.self_attn = torch.nn.MultiheadAttention(
            conv_dim, nhead, batch_first=batch_first
        )
        # Implementation of Feedforward model
        self.linear1 = torch.nn.Linear(conv_dim, dim_feedforward)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear2 = torch.nn.Linear(dim_feedforward, conv_dim)

        # Implementation of gates
        self.gate_linear = torch.nn.Linear(conv_dim, 1, bias=False)
        self.gate_act = torch.nn.Sigmoid()

        self.norm_first = norm_first
        self.use_layer_norm = use_layer_norm

        if self.use_layer_norm:
            self.norm1 = torch.nn.LayerNorm(conv_dim, eps=layer_norm_eps)
            self.norm2 = torch.nn.LayerNorm(conv_dim, eps=layer_norm_eps)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    # self-attention block
    def _sa_block(
        self, x: Tensor, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]
    ) -> Tensor:
        key_padding_mask = ~key_padding_mask.bool()
        x = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
        )[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        g = self.gate_act(self.gate_linear(x))
        h = self.linear1(x)
        h = h * g  # add gate
        h = self.linear2(self.dropout(self.activation(h)))
        return self.dropout2(h)

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = torch.nn.functional.relu
        super().__setstate__(state)

    def forward(
        self, x, src_mask=None, src_key_padding_mask=None, is_causal=None, **kwargs
    ) -> Tensor:
        r"""
        Args:
            x (Tensor): Input of shape :math:`(N, S, H)`.
            src_mask (Tensor, optional): Additive attention mask :math:`(S, S)`. Default: ``None``.
            src_key_padding_mask (Tensor, optional): Padding mask :math:`(N, S)`;
                ``True`` positions are attended to. Default: ``None``.
            is_causal: Unused; present for API compatibility.

        Returns:
            Tensor: Same shape as input.
        """

        if self.use_layer_norm:
            if self.norm_first:
                x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
                x = x + self._ff_block(self.norm2(x))
            else:
                x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
                x = self.norm2(x + self._ff_block(x))

        else:  # do not use layer norm
            x = x + self._sa_block(x, src_mask, src_key_padding_mask)
            x = x + self._ff_block(x)
        return x
