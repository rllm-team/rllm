from __future__ import annotations

import functools

import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel


@functools.cache
def gqa_is_supported() -> bool:
    """Return whether the current PyTorch/CUDA runtime supports efficient GQA."""
    if not torch.cuda.is_available():
        return False

    has_enable_gqa = torch.__version__ >= "2.5"
    if not has_enable_gqa:
        return False

    device = torch.cuda.current_device()
    nvidia_compute_capability = torch.cuda.get_device_capability(device)
    return nvidia_compute_capability[0] >= 8


class Attention(nn.Module):
    """Base projected scaled-dot-product attention module."""

    def __init__(
        self,
        embedding_size: int,
        num_heads: int,
        head_dim: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        linear_kwargs = {"device": device, "dtype": dtype, "bias": False}
        self.q_projection = nn.Linear(
            embedding_size, self.num_heads * self.head_dim, **linear_kwargs
        )
        self.k_projection = nn.Linear(
            embedding_size, self.num_heads * self.head_dim, **linear_kwargs
        )
        self.v_projection = nn.Linear(
            embedding_size, self.num_heads * self.head_dim, **linear_kwargs
        )
        self.out_projection = nn.Linear(
            self.num_heads * self.head_dim, embedding_size, **linear_kwargs
        )
        torch.nn.init.xavier_uniform_(self.q_projection.weight)
        torch.nn.init.xavier_uniform_(self.k_projection.weight)
        torch.nn.init.xavier_uniform_(self.v_projection.weight)
        torch.nn.init.zeros_(self.out_projection.weight)

    def forward(
        self,
        query_input: torch.Tensor,
        key_value_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply attention from ``query_input`` to ``key_value_input``.

        If ``key_value_input`` is omitted, this is standard self-attention.
        Inputs are expected to have shape ``[batch, sequence, embedding_size]``.
        """

        if key_value_input is None:
            key_value_input = query_input

        batch_size, query_length, _ = query_input.shape
        key_length = key_value_input.shape[1]

        query_flat = self.q_projection(query_input)
        key_flat = self.k_projection(key_value_input)
        value_flat = self.v_projection(key_value_input)

        query = query_flat.view(
            batch_size,
            query_length,
            self.num_heads,
            self.head_dim,
        )
        key = key_flat.view(batch_size, key_length, self.num_heads, self.head_dim)
        value = value_flat.view(batch_size, key_length, self.num_heads, self.head_dim)

        output = _batched_scaled_dot_product_attention(query, key, value)
        output = output.reshape(
            batch_size,
            query_length,
            self.num_heads * self.head_dim,
        )
        return self.out_projection(output)


def _batched_scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    """Apply SDPA to tensors shaped ``[batch, sequence, heads, head_dim]``."""

    query_heads = query.permute(0, 2, 1, 3)
    key_heads = key.permute(0, 2, 1, 3)
    value_heads = value.permute(0, 2, 1, 3)

    dtype_supports_gqa = query_heads.dtype in {torch.float16, torch.bfloat16}
    if gqa_is_supported() and dtype_supports_gqa:
        keys = key_heads
        values = value_heads
        enable_gqa = {"enable_gqa": True}
    else:
        keys = key_heads.expand(-1, query_heads.shape[-3], -1, -1)
        values = value_heads.expand(-1, query_heads.shape[-3], -1, -1)
        enable_gqa = {}

    backends = [
        SDPBackend.FLASH_ATTENTION,
        SDPBackend.EFFICIENT_ATTENTION,
        SDPBackend.CUDNN_ATTENTION,
        SDPBackend.MATH,
    ]
    num_parallel_calls = query_heads.shape[:2].numel()
    cuda_max_grid = 65536
    num_iterations = (num_parallel_calls + cuda_max_grid - 1) // cuda_max_grid
    sub_batch = (query_heads.shape[0] + num_iterations - 1) // num_iterations

    with sdpa_kernel(backends=backends):
        outputs = []
        for i in range(num_iterations):
            outputs.append(
                torch.nn.functional.scaled_dot_product_attention(
                    query_heads[i * sub_batch : (i + 1) * sub_batch],
                    keys[i * sub_batch : (i + 1) * sub_batch],
                    values[i * sub_batch : (i + 1) * sub_batch],
                    attn_mask=None,
                    **enable_gqa,
                )
            )
    output = outputs[0] if len(outputs) == 1 else torch.cat(outputs)
    return output.permute(0, 2, 1, 3)
