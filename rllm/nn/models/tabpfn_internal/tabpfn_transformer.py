from __future__ import annotations

import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from .tabpfn_utils import gqa_is_supported


class AddThinkingRows(nn.Module):
    """Prepends learned thinking rows before the actual table rows."""

    def __init__(self, num_thinking_rows: int, embedding_size: int) -> None:
        super().__init__()
        self.num_thinking_rows = int(num_thinking_rows)
        self.row_token_values_TE = nn.Parameter(
            torch.empty(self.num_thinking_rows, embedding_size)
        )
        self.reset_parameters()

    def forward(
        self,
        x_BRiCE: torch.Tensor,
        single_eval_pos: int,
    ) -> tuple[torch.Tensor, int]:
        batch_size, _, num_features, _ = x_BRiCE.shape
        thinking_rows_BTCE = (
            self.row_token_values_TE.unsqueeze(0)
            .unsqueeze(2)
            .expand(batch_size, -1, num_features, -1)
        )
        x_BRCE = torch.cat([thinking_rows_BTCE, x_BRiCE], dim=1)
        return x_BRCE, single_eval_pos + self.num_thinking_rows

    def reset_parameters(self) -> None:
        torch.nn.init.normal_(self.row_token_values_TE)


class Attention(nn.Module):
    """Base attention module used by the official-style transformer blocks."""

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

    def forward(self, x_BrCE: torch.Tensor) -> torch.Tensor:
        br, c, _ = x_BrCE.shape
        q_flat = self.q_projection(x_BrCE)
        k_flat = self.k_projection(x_BrCE)
        v_flat = self.v_projection(x_BrCE)
        q = q_flat.view(br, c, -1, self.head_dim)
        k = k_flat.view(br, c, -1, self.head_dim)
        v = v_flat.view(br, c, -1, self.head_dim)
        out = _batched_scaled_dot_product_attention(q, k, v)
        out = out.reshape(br, c, self.num_heads * self.head_dim)
        return self.out_projection(out)


class AlongColumnAttention(Attention):
    """Attention across rows within a single feature group."""

    def forward(
        self,
        x_BcRE: torch.Tensor,
        single_eval_pos: int | None = None,
    ) -> torch.Tensor:
        bc, r, _ = x_BcRE.shape
        n_train = r if single_eval_pos is None else int(single_eval_pos)

        q_flat = self.q_projection(x_BcRE)
        k_flat = self.k_projection(x_BcRE[:, :n_train])
        v_flat = self.v_projection(x_BcRE[:, :n_train])
        q = q_flat.view(bc, r, -1, self.head_dim)
        k = k_flat.view(bc, n_train, -1, self.head_dim)
        v = v_flat.view(bc, n_train, -1, self.head_dim)

        if single_eval_pos == r:
            out = _batched_scaled_dot_product_attention(q, k, v)
        else:
            out_train = _batched_scaled_dot_product_attention(q[:, :n_train], k, v)
            out_test = _batched_scaled_dot_product_attention(
                q[:, n_train:],
                k[:, :, :1],
                v[:, :, :1],
            )
            out = torch.cat([out_train, out_test], dim=1)

        out = out.reshape(bc, r, self.num_heads * self.head_dim)
        return self.out_projection(out)


def _batched_scaled_dot_product_attention(
    q_BSHD: torch.Tensor,
    k_BSJD: torch.Tensor,
    v_BSJD: torch.Tensor,
) -> torch.Tensor:
    q_BHSD = q_BSHD.permute(0, 2, 1, 3)
    k_BJSD = k_BSJD.permute(0, 2, 1, 3)
    v_BJSD = v_BSJD.permute(0, 2, 1, 3)

    dtype_supports_gqa = q_BHSD.dtype in {torch.float16, torch.bfloat16}
    if gqa_is_supported() and dtype_supports_gqa:
        keys = k_BJSD
        values = v_BJSD
        enable_gqa = {"enable_gqa": True}
    else:
        keys = k_BJSD.expand(-1, q_BHSD.shape[-3], -1, -1)
        values = v_BJSD.expand(-1, q_BHSD.shape[-3], -1, -1)
        enable_gqa = {}

    backends = [
        SDPBackend.FLASH_ATTENTION,
        SDPBackend.EFFICIENT_ATTENTION,
        SDPBackend.CUDNN_ATTENTION,
        SDPBackend.MATH,
    ]
    num_parallel_calls = q_BHSD.shape[:2].numel()
    cuda_max_grid = 65536
    num_iterations = (num_parallel_calls + cuda_max_grid - 1) // cuda_max_grid
    sub_batch = (q_BHSD.shape[0] + num_iterations - 1) // num_iterations

    with sdpa_kernel(backends=backends):
        outputs = []
        for i in range(num_iterations):
            outputs.append(
                torch.nn.functional.scaled_dot_product_attention(
                    q_BHSD[i * sub_batch : (i + 1) * sub_batch],
                    keys[i * sub_batch : (i + 1) * sub_batch],
                    values[i * sub_batch : (i + 1) * sub_batch],
                    attn_mask=None,
                    **enable_gqa,
                )
            )
    output_BHSD = outputs[0] if len(outputs) == 1 else torch.cat(outputs)
    return output_BHSD.permute(0, 2, 1, 3)


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


class TabPFNBlock(nn.Module):
    """One TabPFN transformer block."""

    def __init__(
        self,
        *,
        emsize: int,
        nhead: int,
        dim_feedforward: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        head_dim = emsize // nhead
        self.per_sample_attention_between_features = AlongRowAttention(
            embedding_size=emsize,
            num_heads=nhead,
            head_dim=head_dim,
            device=device,
            dtype=dtype,
        )
        self.per_column_attention_between_cells = AlongColumnAttention(
            embedding_size=emsize,
            num_heads=nhead,
            head_dim=head_dim,
            device=device,
            dtype=dtype,
        )
        norm_kwargs = {"device": device, "dtype": dtype, "elementwise_affine": True}
        self.layernorm_mha1 = LowerPrecisionRMSNorm(emsize, **norm_kwargs)
        self.layernorm_mha2 = LowerPrecisionRMSNorm(emsize, **norm_kwargs)
        self.layernorm_mlp = LowerPrecisionRMSNorm(emsize, **norm_kwargs)
        self.mlp = nn.Sequential(
            nn.Linear(emsize, dim_feedforward, bias=False, device=device, dtype=dtype),
            nn.GELU(),
            nn.Linear(dim_feedforward, emsize, bias=False, device=device, dtype=dtype),
        )
        torch.nn.init.zeros_(self.mlp[2].weight)

    def forward(
        self,
        x_BRCE: torch.Tensor,
        single_eval_pos: int,
        save_peak_memory_factor: int | None = None,
    ) -> torch.Tensor:
        del save_peak_memory_factor
        b, r, c, e = x_BRCE.shape

        x_rows = x_BRCE.reshape(b * r, c, e)
        x_rows = x_rows + self.per_sample_attention_between_features(x_rows)
        x_rows = self.layernorm_mha1(x_rows)
        x_BRCE = x_rows.view(b, r, c, e)

        x_cols = x_BRCE.transpose(1, 2).reshape(b * c, r, e)
        x_cols = x_cols + self.per_column_attention_between_cells(
            x_cols,
            single_eval_pos=single_eval_pos,
        )
        x_cols = self.layernorm_mha2(x_cols)
        x_BRCE = x_cols.view(b, c, r, e).transpose(1, 2).contiguous()

        x_BRCE = x_BRCE + self.mlp(x_BRCE)
        return self.layernorm_mlp(x_BRCE)
