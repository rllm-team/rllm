from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
from torch import nn

from rllm.nn.attention import (
    Attention,
    _batched_scaled_dot_product_attention,
)
from rllm.nn.encoder.tabpfn_pre_encoder import TabPFNXPreEncoder, TabPFNYPreEncoder


class LowerPrecisionRMSNorm(nn.RMSNorm):
    """RMSNorm variant used by the retained TabPFN runtime.

    Small hidden states are normalized outside autocast to match the checkpoint
    behavior while still allowing low-precision tensors through the rest of the
    model.
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if (
            input.dtype in (torch.float16, torch.bfloat16)
            and sum(self.normalized_shape) < 512
        ):
            with torch.amp.autocast("cuda" if input.is_cuda else "cpu", enabled=False):
                return super().forward(input)
        return super().forward(input)


class AddThinkingRows(nn.Module):
    """Prepend learned context rows to the embedded table.

    TabPFN v2.6 uses a fixed number of trainable rows before the observed table
    rows. These rows are shared across feature groups and are included in the
    transformer context, so ``single_eval_pos`` must be shifted by the same
    amount.

    Args:
        num_thinking_rows (int): Number of learned rows to prepend.
        embedding_size (int): Size of each row embedding.
    """

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


class AlongColumnAttention(Attention):
    """Attention over table rows for each feature group.

    During prediction, training rows attend to the full training context, while
    test rows attend only to the shared prefix used by the retained checkpoint.
    The input is flattened over batch and feature-group dimensions.
    """

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


class TabPFNLayer(nn.Module):
    """Single TabPFN transformer block.

    Each block alternates attention across feature groups within a row, attention
    across rows within each feature group, and a feed-forward network. The tensor
    layout is kept as ``[batch, rows, feature_groups, embedding]`` between
    blocks.

    Args:
        emsize (int): Embedding size used by the retained checkpoint.
        nhead (int): Number of attention heads.
        dim_feedforward (int): Hidden dimension of the feed-forward network.
        device (torch.device | str, optional): Device for module parameters.
        dtype (torch.dtype, optional): Parameter dtype.
    """

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
        self.per_sample_attention_between_features = Attention(
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


class TabPFNTransformer(nn.Module):
    """Transformer stack used inside the TabPFN backbone.

    This module contains only the repeated transformer layers. Feature encoding,
    target encoding, learned thinking rows, and output projection live in
    :class:`TabPFNBackbone`.

    Args:
        emsize (int): Embedding size used throughout the transformer.
        nlayers (int): Number of transformer blocks.
        nhead (int): Number of attention heads per block.
        device (torch.device | str, optional): Device for module parameters.
        dtype (torch.dtype, optional): Parameter dtype.
    """

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


class TabPFNBackbone(nn.Module):
    """Retained TabPFN v2.6 backbone runtime.

    This module wraps the complete TabPFN backbone path: feature and target
    encoders, learned thinking rows, the :class:`TabPFNTransformer` stack, and
    the final projection from target-channel embeddings. It is designed for
    loading retained TabPFN checkpoints rather than for training a new
    architecture from scratch.

    Args:
        emsize (int): Embedding size used by the checkpoint.
        nlayers (int): Number of transformer blocks.
        nhead (int): Number of attention heads.
        features_per_group (int): Number of input features packed into a group.
        num_thinking_rows (int): Number of learned context rows to prepend.
        encoder_type (str): Feature encoder type, either ``"linear"`` or ``"mlp"``.
        encoder_mlp_hidden_dim (int): Hidden size for the MLP feature encoder.
        n_out (int): Output dimension of the prediction head.
        task_type (str): Task type, either ``"multiclass"`` or ``"regression"``.
        column_embeddings_path (str | Path): Path to pregenerated column embeddings.
        device (torch.device | str, optional): Device for module parameters.
        dtype (torch.dtype, optional): Parameter dtype.
    """

    def __init__(
        self,
        *,
        emsize: int,
        nlayers: int,
        nhead: int,
        features_per_group: int,
        num_thinking_rows: int,
        encoder_type: Literal["linear", "mlp"],
        encoder_mlp_hidden_dim: int,
        n_out: int,
        task_type: Literal["multiclass", "regression"],
        column_embeddings_path: str | Path,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.n_out = int(n_out)
        self.task_type = task_type
        self.input_size = int(emsize)
        self.hidden_size = self.input_size * 2
        self.features_per_group = int(features_per_group)
        self.cache_trainset_representation = False
        self.ninp = self.input_size
        self.feature_positional_embedding = "subspace"
        self._do_encoder_nan_check = True

        self.x_pre_encoder = TabPFNXPreEncoder(
            emsize=self.input_size,
            features_per_group=self.features_per_group,
            encoder_type=encoder_type,
            encoder_mlp_hidden_dim=encoder_mlp_hidden_dim,
            column_embeddings_path=column_embeddings_path,
            device=device,
            dtype=dtype,
        )
        self.y_pre_encoder = TabPFNYPreEncoder(
            emsize=self.input_size,
            task_type=task_type,
            device=device,
            dtype=dtype,
        )
        self.add_thinking_rows = AddThinkingRows(
            num_thinking_rows=num_thinking_rows,
            embedding_size=self.input_size,
        )
        self.transformer_encoder = TabPFNTransformer(
            emsize=emsize,
            nlayers=nlayers,
            nhead=nhead,
            device=device,
            dtype=dtype,
        )
        self.output_projection = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size, device=device, dtype=dtype),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.n_out, device=device, dtype=dtype),
        )

    def preprocess_embeddings(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None,
        *,
        single_eval_pos: int | None = None,
    ) -> dict[str, torch.Tensor]:
        if y is None:
            y = torch.zeros(0, device=x.device, dtype=x.dtype)
        if y.ndim == 1:
            y = y[:, None]

        num_rows, batch_size, *_ = x.shape
        num_train_labels = (
            int(single_eval_pos) if single_eval_pos is not None else y.shape[0]
        )
        embedded_x = self.x_pre_encoder(
            x,
            single_eval_pos=num_train_labels,
        )
        embedded_y = self.y_pre_encoder(
            y,
            num_train_rows=num_rows,
            num_train_labels=num_train_labels,
            batch_size=batch_size,
        )
        embedded_input = torch.cat((embedded_x, embedded_y[:, :, None]), dim=2)
        return {
            "embedded_x": embedded_x,
            "embedded_y": embedded_y,
            "embedded_input": embedded_input,
        }

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None,
        *,
        single_eval_pos: int | None = None,
        only_return_standard_out: bool = True,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        if isinstance(x, dict):
            x = x["main"]
        if isinstance(y, dict):
            y = y["main"]
        if y is None:
            y = torch.zeros(0, device=x.device, dtype=x.dtype)

        if (
            not self.training
            and self.task_type == "multiclass"
            and (y > self.n_out - 1).any()
        ):
            raise ValueError(
                "Target is out of range. Make sure to use an ordinal encoded target. "
                f"Expected target values between 0 and {self.n_out - 1}, but got values"
                f" greater than {self.n_out - 1}."
            )

        num_rows = x.shape[0]
        num_train_labels = (
            int(single_eval_pos) if single_eval_pos is not None else y.shape[0]
        )
        if y.shape[0] != num_train_labels and y.shape[0] != 0:
            raise ValueError(
                "Mismatch between provided y rows and single_eval_pos: "
                f"y.shape[0]={y.shape[0]}, single_eval_pos={num_train_labels}."
            )

        encoded = self.preprocess_embeddings(
            x,
            y,
            single_eval_pos=num_train_labels,
        )
        embedded_input = encoded["embedded_input"]
        if self._do_encoder_nan_check:
            if torch.isnan(embedded_input).any():
                raise ValueError(
                    "Found NaNs in the encoded x and y. Make sure to use "
                    "a NaN-handling encoder."
                )
            self._do_encoder_nan_check = False

        hidden, num_train_and_thinking_rows = self.add_thinking_rows(
            embedded_input,
            single_eval_pos=num_train_labels,
        )
        hidden = self.transformer_encoder(
            hidden, single_eval_pos=num_train_and_thinking_rows
        )

        test_embeddings = hidden[:, num_train_and_thinking_rows:, -1]
        test_embeddings = test_embeddings.transpose(0, 1)
        output = self.output_projection(test_embeddings)

        if only_return_standard_out:
            return output

        train_rows_start = self.add_thinking_rows.num_thinking_rows
        train_rows_end = min(
            num_train_and_thinking_rows,
            hidden.shape[1],
            num_rows + train_rows_start,
        )
        train_embeddings = hidden[:, train_rows_start:train_rows_end, -1]
        train_embeddings = train_embeddings.transpose(0, 1)

        return {
            "standard": output,
            "embedded_x": encoded["embedded_x"],
            "embedded_y": encoded["embedded_y"],
            "embedded_input": embedded_input,
            "train_embeddings": train_embeddings,
            "test_embeddings": test_embeddings,
        }
