#  Copyright (c) Prior Labs GmbH 2025.

from __future__ import annotations

import math
from functools import partial
from typing_extensions import override

import torch
from torch.utils.checkpoint import checkpoint


class MultiHeadAttention(torch.nn.Module):
    def newly_initialized_input_weight(
        self,
        dims: list[int],
        nhead: int,
        device: torch.device | None,
        dtype: torch.dtype | None,
    ) -> torch.nn.Parameter:
        w = torch.nn.Parameter(torch.empty(*dims, device=device, dtype=dtype))
        d, input_size = dims[-2:]
        std = math.sqrt(2.0 / float(nhead * d + input_size)) * self.init_gain
        a = math.sqrt(3.0) * std
        torch.nn.init.uniform_(w, -a, a)
        return w

    def __init__(  # noqa: PLR0913
        self,
        *,
        input_size: int,
        output_size: int,
        d_k: int,
        d_v: int,
        nhead: int,
        device: torch.device | None,
        dtype: torch.dtype | None,
        share_kv_across_n_heads: int = 1,  # 保留参数以兼容，但必须为 1
        dropout_p: float | None = None,
        softmax_scale: float | None = None,
        initialize_output_to_zero: bool = False,
        recompute: bool = False,
        init_gain: float = 1.0,
    ):
        super().__init__()
        if share_kv_across_n_heads != 1:
            raise ValueError(
                "Only share_kv_across_n_heads=1 is supported with merged weight scheme"
            )
        if d_k != d_v:
            raise ValueError("Only d_k == d_v is supported with merged weight scheme")

        self._input_size = input_size
        self._output_size = output_size
        self._d_k = d_k
        self._d_v = d_v
        self._nhead = nhead
        self._device = device
        self._dtype = dtype
        self.dropout_p = dropout_p
        self.softmax_scale = softmax_scale
        self.recompute = recompute
        self.init_gain = init_gain

        w_out = torch.nn.Parameter(
            torch.empty(nhead, d_v, output_size, device=device, dtype=dtype),
        )
        if initialize_output_to_zero:
            torch.nn.init.zeros_(w_out)
        else:
            torch.nn.init.xavier_uniform_(w_out)

        w_qkv = self.newly_initialized_input_weight(
            [3, self._nhead, self._d_k, self._input_size],
            nhead=self._nhead,
            device=device,
            dtype=dtype,
        )
        self.register_parameter("_w_out", w_out)
        self.register_parameter("_w_qkv", w_qkv)

        if recompute:
            self.forward = partial(checkpoint, self.forward, use_reentrant=False)  # type: ignore

    @override
    def forward(
        self,
        x: torch.Tensor,
        x_kv: torch.Tensor | None = None,
        *,
        add_input: bool = False,
        allow_inplace: bool = False,
        reuse_first_head_kv: bool = False,
    ):
        """X is the current hidden and has a shape of [batch, ..., seq_len, input_size].
        If 'x_kv' is not None, keys and values are obtained by applying the
        respective linear transformations to 'x_kv'.
        Else, keys and values are attained by applying the respective linear
        transformations to 'x' (self attention).
        """

        # The attention computation expects the input to have shape [batch * ..., seq_len, input_size].
        x_shape_after_transpose = x.shape
        x = x.reshape(-1, *x.shape[-2:])
        residual = x
        if x_kv is not None:
            x_kv = x_kv.reshape(-1, *x_kv.shape[-2:])

        output: torch.Tensor = self._compute(
            x,
            x_kv,
            reuse_first_head_kv=reuse_first_head_kv,
        )

        if add_input:
            output = residual.add_(output) if allow_inplace else residual + output

        return output.reshape(x_shape_after_transpose[:-1] + output.shape[-1:])

    def _compute(
        self,
        x: torch.Tensor,
        x_kv: torch.Tensor | None,
        *,
        reuse_first_head_kv: bool = False,
    ) -> torch.Tensor:
        """Attention computation.
        Called by 'forward', potentially on shards, once shapes have been normalized.
        """
        # compatible with cross-attention
        if x_kv is None:
            x_kv = x

        # Self-attention fast path: compute q, k, v together from merged weight.
        if x is x_kv:
            q = k = v = kv = None
            qkv = torch.einsum("... s, j h d s -> ... j h d", x, self._w_qkv)
        else:
            # Cross-attention path: split merged weight into q and kv projections.
            w_q = self._w_qkv[0]
            w_kv = self._w_qkv[1:]

            q = torch.einsum("... s, h d s -> ... h d", x, w_q)
            k = v = qkv = None
            if reuse_first_head_kv:
                kv_first = torch.einsum(
                    "... s, j h d s -> ... j h d",
                    x_kv,
                    w_kv[:, :1],
                )
                expand_shape = list(kv_first.shape)
                expand_shape[-2] = self._nhead
                kv = kv_first.expand(*expand_shape)
            else:
                kv = torch.einsum("... s, j h d s -> ... j h d", x_kv, w_kv)

        if qkv is not None:
            q, k, v = qkv.unbind(dim=-3)
        elif kv is not None:
            k, v = kv.unbind(dim=-3)

        if q is None or k is None or v is None:
            raise ValueError("q, k, v must be provided via qkv or (q and (kv or k,v)).")

        _, _, nhead, d_k = q.shape
        _, _, nhead_kv, _ = v.shape
        dropout_p = 0.0 if self.dropout_p is None else self.dropout_p

        # Standard multi-head attention without GQA
        if nhead != nhead_kv:
            raise ValueError(
                f"head count mismatch (nhead={nhead}, nhead_kv={nhead_kv}). "
                "Only standard multi-head attention is supported."
            )

        extra_inputs = {}
        if self.softmax_scale is not None:
            extra_inputs["scale"] = self.softmax_scale

        attention_head_outputs = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            dropout_p=dropout_p,
            **extra_inputs,
        ).transpose(1, 2)

        return torch.einsum(
            "... h d, h d s -> ... s",
            attention_head_outputs,
            self._w_out,
        )
