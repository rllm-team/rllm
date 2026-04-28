from __future__ import annotations

import torch
from torch import nn

from rllm.nn.attention import (
    AlongColumnAttention,
    AlongRowAttention,
    LowerPrecisionRMSNorm,
)


class AddThinkingRows(nn.Module):
    """Prepends learned thinking rows before the actual table rows."""

    def __init__(self, num_thinking_rows: int, embedding_size: int) -> None:
        super().__init__()
        self.num_thinking_rows = int(num_thinking_rows)
        self.row_token_values = nn.Parameter(
            torch.empty(self.num_thinking_rows, embedding_size)
        )
        self.reset_parameters()

    def forward(
        self,
        embedded_input: torch.Tensor,
        single_eval_pos: int,
    ) -> tuple[torch.Tensor, int]:
        batch_size, _, num_features, _ = embedded_input.shape
        thinking_rows = (
            self.row_token_values.unsqueeze(0)
            .unsqueeze(2)
            .expand(batch_size, -1, num_features, -1)
        )
        output = torch.cat([thinking_rows, embedded_input], dim=1)
        return output, single_eval_pos + self.num_thinking_rows

    def reset_parameters(self) -> None:
        torch.nn.init.normal_(self.row_token_values)


class TabPFNLayer(nn.Module):
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
        x: torch.Tensor,
        single_eval_pos: int,
    ) -> torch.Tensor:
        batch_size, num_rows, num_feature_groups, embedding_size = x.shape

        # Row attention
        x = x.reshape(
            batch_size * num_rows,
            num_feature_groups,
            embedding_size,
        )
        x = x + self.per_sample_attention_between_features(x)
        x_row_attn = self.layernorm_mha1(x)
        x_row_attn = x_row_attn.view(
            batch_size,
            num_rows,
            num_feature_groups,
            embedding_size,
        )

        # Column attention
        x_row_attn = x_row_attn.transpose(1, 2).reshape(
            batch_size * num_feature_groups,
            num_rows,
            embedding_size,
        )
        x_col_attn = x_row_attn + self.per_column_attention_between_cells(
            x_row_attn,
            single_eval_pos=single_eval_pos,
        )
        x_col_attn = self.layernorm_mha2(x_col_attn)
        x = (
            x_col_attn.view(
                batch_size,
                num_feature_groups,
                num_rows,
                embedding_size,
            )
            .transpose(1, 2)
            .contiguous()
        )

        # FFN
        x = x + self.mlp(x)
        return self.layernorm_mlp(x)


class TabPFNBackbone(nn.Module):
    """Pure transformer backbone used by the TabPFN model."""

    def __init__(
        self,
        *,
        emsize: int,
        nlayers: int,
        nhead: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.input_size = int(emsize)
        self.hidden_size = self.input_size * 2
        self.blocks = nn.ModuleList(
            TabPFNLayer(
                emsize=self.input_size,
                nhead=nhead,
                dim_feedforward=self.hidden_size,
                device=device,
                dtype=dtype,
            )
            for _ in range(nlayers)
        )

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        single_eval_pos: int,
    ) -> torch.Tensor:
        for block in self.blocks:
            hidden = block(
                hidden,
                single_eval_pos=single_eval_pos,
            )
        return hidden
