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
    """Base attention module used by official-style transformer blocks."""

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


class AlongRowAttention(Attention):
    """Attention across feature groups within a single row."""

    def forward(self, row_input: torch.Tensor) -> torch.Tensor:
        batch_rows, num_feature_groups, _ = row_input.shape
        query_flat = self.q_projection(row_input)
        key_flat = self.k_projection(row_input)
        value_flat = self.v_projection(row_input)
        query = query_flat.view(
            batch_rows,
            num_feature_groups,
            -1,
            self.head_dim,
        )
        key = key_flat.view(batch_rows, num_feature_groups, -1, self.head_dim)
        value = value_flat.view(batch_rows, num_feature_groups, -1, self.head_dim)
        output = _batched_scaled_dot_product_attention(query, key, value)
        output = output.reshape(
            batch_rows,
            num_feature_groups,
            self.num_heads * self.head_dim,
        )
        return self.out_projection(output)


class AlongColumnAttention(Attention):
    """Attention across rows within a single feature group."""

    def forward(
        self,
        column_input: torch.Tensor,
        single_eval_pos: int | None = None,
    ) -> torch.Tensor:
        batch_columns, num_rows, _ = column_input.shape
        num_train_rows = num_rows if single_eval_pos is None else int(single_eval_pos)

        query_flat = self.q_projection(column_input)
        key_flat = self.k_projection(column_input[:, :num_train_rows])
        value_flat = self.v_projection(column_input[:, :num_train_rows])
        query = query_flat.view(batch_columns, num_rows, -1, self.head_dim)
        key = key_flat.view(batch_columns, num_train_rows, -1, self.head_dim)
        value = value_flat.view(batch_columns, num_train_rows, -1, self.head_dim)

        if single_eval_pos == num_rows:
            output = _batched_scaled_dot_product_attention(query, key, value)
        else:
            train_output = _batched_scaled_dot_product_attention(
                query[:, :num_train_rows],
                key,
                value,
            )
            test_output = _batched_scaled_dot_product_attention(
                query[:, num_train_rows:],
                key[:, :, :1],
                value[:, :, :1],
            )
            output = torch.cat([train_output, test_output], dim=1)

        output = output.reshape(batch_columns, num_rows, self.num_heads * self.head_dim)
        return self.out_projection(output)


def _batched_scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
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


class LowerPrecisionRMSNorm(nn.RMSNorm):
    """RMSNorm variant that keeps low precision in autocast mode."""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if (
            input.dtype in (torch.float16, torch.bfloat16)
            and sum(self.normalized_shape) < 512
        ):
            with torch.amp.autocast("cuda" if input.is_cuda else "cpu", enabled=False):
                return super().forward(input)
        return super().forward(input)
