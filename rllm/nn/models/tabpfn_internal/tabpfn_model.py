from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
from torch import nn

from rllm.nn.encoder.tabpfn_pre_encoder import TabPFNXPreEncoder, TabPFNYPreEncoder
from .tabpfn_utils import TorchStandardScaler

from .tabpfn_backbone import TabPFNBackbone, TabPFNConfig
from .tabpfn_transformer import AddThinkingRows


class TabPFNModel(nn.Module):
    """Full TabPFN model built around a pure transformer backbone."""

    def __init__(
        self,
        *,
        config: TabPFNConfig,
        n_out: int,
        task_type: Literal["multiclass", "regression"],
        column_embeddings_path: str | Path,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.n_out = int(n_out)
        self.task_type = task_type
        self.input_size = int(config.emsize)
        self.hidden_size = self.input_size * 2
        self.features_per_group = int(config.features_per_group)
        self.cache_trainset_representation = False
        self.ninp = self.input_size
        self.feature_positional_embedding = "subspace"
        self._do_encoder_nan_check = True

        self.x_pre_encoder = TabPFNXPreEncoder(
            emsize=self.input_size,
            features_per_group=self.features_per_group,
            encoder_type=config.encoder_type,
            encoder_mlp_hidden_dim=config.encoder_mlp_hidden_dim,
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
            num_thinking_rows=config.num_thinking_rows,
            embedding_size=self.input_size,
        )
        self.backbone = TabPFNBackbone(
            config=config,
            device=device,
            dtype=dtype,
        )
        self.output_projection = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size, device=device, dtype=dtype),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.n_out, device=device, dtype=dtype),
        )

    @property
    def pre_generated_column_embeddings(self) -> torch.Tensor:
        return self.x_pre_encoder.pre_generated_column_embeddings

    @pre_generated_column_embeddings.setter
    def pre_generated_column_embeddings(self, value: torch.Tensor) -> None:
        self.x_pre_encoder.pre_generated_column_embeddings = value

    @property
    def standard_scaler(self) -> object:
        return TorchStandardScaler()

    @property
    def feature_group_embedder(self) -> nn.Module:
        return self.x_pre_encoder.feature_group_embedder

    @property
    def target_embedder(self) -> nn.Module:
        return self.y_pre_encoder.target_embedder

    @property
    def blocks(self) -> nn.ModuleList:
        return self.backbone.blocks

    def _add_column_embeddings(self, x_BRGX: torch.Tensor) -> torch.Tensor:
        return self.x_pre_encoder._add_column_embeddings(x_BRGX)

    def _preprocess_and_embed_features(
        self,
        x_RiBC: torch.Tensor,
        num_train_labels: int,
        batch_size: int,
    ) -> torch.Tensor:
        return self.x_pre_encoder._preprocess_and_embed_features(
            x_RiBC=x_RiBC,
            num_train_labels=num_train_labels,
            batch_size=batch_size,
        )

    def _preprocess_and_embed_targets(
        self,
        y: torch.Tensor,
        num_train_rows: int,
        num_train_labels: int,
        batch_size: int,
    ) -> torch.Tensor:
        return self.y_pre_encoder._preprocess_and_embed_targets(
            y=y,
            num_train_rows=num_train_rows,
            num_train_labels=num_train_labels,
            batch_size=batch_size,
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
        categorical_inds: list[list[int]] | None = None,
        performance_options: object | None = None,
        task_type: str | None = None,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        del categorical_inds, performance_options, task_type

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
        num_train_labels = int(single_eval_pos) if single_eval_pos is not None else y.shape[0]
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
        hidden = self.backbone(hidden, single_eval_pos=num_train_and_thinking_rows)

        test_embeddings_BMD = hidden[:, num_train_and_thinking_rows:, -1]
        test_embeddings_MBD = test_embeddings_BMD.transpose(0, 1)
        test_output_MB1 = self.output_projection(test_embeddings_MBD)

        if only_return_standard_out:
            return test_output_MB1

        train_rows_start = self.add_thinking_rows.num_thinking_rows
        train_rows_end = min(num_train_and_thinking_rows, hidden.shape[1], num_rows + train_rows_start)
        train_embeddings_BND = hidden[:, train_rows_start:train_rows_end, -1]
        train_embeddings_NBD = train_embeddings_BND.transpose(0, 1)

        return {
            "standard": test_output_MB1,
            "embedded_x": encoded["embedded_x"],
            "embedded_y": encoded["embedded_y"],
            "embedded_input": embedded_input,
            "train_embeddings": train_embeddings_NBD,
            "test_embeddings": test_embeddings_MBD,
        }
