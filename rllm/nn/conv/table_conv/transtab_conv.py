from __future__ import annotations
from typing import Optional

import torch
from torch import Tensor


def _get_activation_fn(activation):
    if activation == "relu":
        return torch.nn.functional.relu
    elif activation == "gelu":
        return torch.nn.functional.gelu
    elif activation == 'selu':
        return torch.nn.functional.selu
    elif activation == 'leakyrelu':
        return torch.nn.functional.leaky_relu
    raise RuntimeError("activation should be relu/gelu/selu/leakyrelu, not {}".format(activation))


class TransTabConv(torch.nn.Module):
    r"""The TransTabConv module introduced in
    `"TransTab: Learning Transferable Tabular Transformers Across Tables"`
    <https://arxiv.org/abs/2205.09328>`_ paper.

    This layer implements a single Transformer encoder block customized for
    tabular transfer learning. It combines multi-head self-attention, a gated
    feedforward network, optional pre-/post-layer normalization, residual
    connections, and dropout to capture complex feature interactions in table
    data.

    Args:
        d_model (int): Dimensionality of input and output feature vectors. (default: required)
        nhead (int): Number of attention heads. (default: required)
        dim_feedforward (int): Hidden dimensionality of the feedforward network.
            If None, defaults to `d_model`. (default: 2048)
        dropout (float): Dropout probability applied in attention and
            feedforward sublayers. (default: 0.1)
        activation (Union[str, Callable]): Activation function for the
            feedforward network, specified as a callable or a string name
            (e.g., "relu"). (default: torch.nn.functional.relu)
        layer_norm_eps (float): Epsilon value for all LayerNorm layers to
            ensure numerical stability. (default: 1e-5)
        batch_first (bool): If True, input and output tensors are expected
            in shape `(batch_size, seq_len, d_model)`; otherwise
            `(seq_len, batch_size, d_model)`. (default: True)
        norm_first (bool): If True, apply LayerNorm before self-attention
            and feedforward; otherwise apply after the residual connection.
            (default: False)
        use_layer_norm (bool): Whether to include LayerNorm layers in each
            sub-block. (default: True)
    """

    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, conv_dim, nhead, dim_feedforward=2048, dropout=0.1, activation=torch.nn.functional.relu,
                 layer_norm_eps=1e-5, batch_first=True, norm_first=False,
                 use_layer_norm=True) -> None:
        super().__init__()
        self.self_attn = torch.nn.MultiheadAttention(conv_dim, nhead, batch_first=batch_first)
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
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        key_padding_mask = ~key_padding_mask.bool()
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           )[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        g = self.gate_act(self.gate_linear(x))
        h = self.linear1(x)
        h = h * g   # add gate
        h = self.linear2(self.dropout(self.activation(h)))
        return self.dropout2(h)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = torch.nn.functional.relu
        super().__setstate__(state)

    def forward(self, x, src_mask=None, src_key_padding_mask=None, is_causal=None, **kwargs) -> Tensor:
        r"""Pass the input through this encoder layer.

        Args:
            src (Tensor): Input tensor of shape
                `(batch_size, seq_len, conv_dim)` if `batch_first=True`,
                else `(seq_len, batch_size, conv_dim)`.
            src_mask (Optional[Tensor]): Attention mask of shape
                `(seq_len, seq_len)` or broadcastable. (default: None)
            src_key_padding_mask (Optional[Tensor]): Padding mask of shape
                `(batch_size, seq_len)` where True values are ignored. (default: None)
            is_causal (Optional[bool]): Unused; present for API compatibility.

        Returns:
            Tensor: Output tensor of the same shape as `src`, after applying
            self-attention, gated feedforward, residual connections, and
            optional layer normalization.
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
