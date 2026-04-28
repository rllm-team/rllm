from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
from torch import nn

from rllm.nn.encoder.col_encoder._feature_group_reshape_encoder import (
    FeatureGroupReshapeEncoder,
)
from rllm.nn.encoder.col_encoder._input_normalization_encoder import (
    InputNormalizationEncoder,
)
from rllm.nn.encoder.col_encoder._nan_inf_indicator_encoder import (
    NanInfIndicatorEncoder,
)
from rllm.nn.encoder.col_encoder._nan_handling_encoder import NanHandlingEncoder
from rllm.nn.encoder.col_encoder._remove_constant_features_encoder import (
    RemoveConstantFeaturesEncoder,
)
from rllm.nn.encoder.col_encoder._variable_num_features_encoder import (
    VariableNumFeaturesEncoder,
)


NAN_INDICATOR = -2.0
INFINITY_INDICATOR = 2.0
NEG_INFINITY_INDICATOR = 4.0
ENCODING_SIZE_MULTIPLIER = 2


def load_column_embeddings(path: str | Path) -> torch.Tensor:
    """Load persisted TabPFN column embeddings from a companion .pt file."""

    col_embedding_path = Path(path)
    return torch.load(col_embedding_path, map_location="cpu", weights_only=True)


class TabPFNXPreEncoder(nn.Module):
    """Preprocess and embed TabPFN feature groups."""

    def __init__(
        self,
        *,
        emsize: int,
        features_per_group: int,
        encoder_type: Literal["linear", "mlp"],
        encoder_mlp_hidden_dim: int,
        column_embeddings_path: str | Path,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.input_size = int(emsize)
        self.features_per_group = int(features_per_group)

        encoding_size = self.features_per_group * ENCODING_SIZE_MULTIPLIER
        if encoder_type == "mlp":
            self.feature_group_embedder = nn.Sequential(
                nn.Linear(
                    encoding_size,
                    encoder_mlp_hidden_dim,
                    bias=False,
                    device=device,
                    dtype=dtype,
                ),
                nn.GELU(),
                nn.Linear(
                    encoder_mlp_hidden_dim,
                    self.input_size,
                    bias=False,
                    device=device,
                    dtype=dtype,
                ),
            )
        else:
            self.feature_group_embedder = nn.Linear(
                encoding_size,
                self.input_size,
                bias=False,
                device=device,
                dtype=dtype,
            )

        self.nan_handling_encoder = NanHandlingEncoder()
        self.input_normalization_encoder = InputNormalizationEncoder(
            normalize_x=True,
            remove_outliers=False,
        )
        self.remove_constant_features_encoder = RemoveConstantFeaturesEncoder()
        self.feature_group_reshape_encoder = FeatureGroupReshapeEncoder(
            num_features_per_group=self.features_per_group,
        )
        self.nan_inf_indicator_encoder = NanInfIndicatorEncoder(
            nan_indicator=NAN_INDICATOR,
            pos_inf_indicator=INFINITY_INDICATOR,
            neg_inf_indicator=NEG_INFINITY_INDICATOR,
        )
        self.variable_num_features_encoder = VariableNumFeaturesEncoder(
            num_features=self.features_per_group,
            normalize_by_used_features=True,
            normalize_by_sqrt=True,
        )
        pre_generated_column_embeddings = load_column_embeddings(
            column_embeddings_path
        ).to(
            device=device,
            dtype=dtype if dtype is not None else torch.float32,
        )
        self.feature_positional_embedding_embeddings = nn.Linear(
            self.input_size // 4,
            self.input_size,
            device=device,
            dtype=dtype,
        )
        self.register_buffer(
            "pre_generated_column_embeddings",
            pre_generated_column_embeddings,
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        single_eval_pos: int,
    ) -> torch.Tensor:
        return self._preprocess_and_embed_features(
            features=x,
            num_train_labels=single_eval_pos,
            batch_size=x.shape[1],
        )

    def _add_column_embeddings(self, grouped_features: torch.Tensor) -> torch.Tensor:
        generator = torch.Generator(device=grouped_features.device).manual_seed(42)
        num_cols, encoding_size = grouped_features.shape[2], grouped_features.shape[3]
        embs = torch.randn(
            (num_cols, encoding_size // 4),
            device=grouped_features.device,
            dtype=grouped_features.dtype,
            generator=generator,
        )
        if (
            self.pre_generated_column_embeddings.numel() > 0
            and embs.shape[1] == self.pre_generated_column_embeddings.shape[1]
        ):
            use_rows = min(
                embs.shape[0],
                self.pre_generated_column_embeddings.shape[0],
            )
            embs[:use_rows] = self.pre_generated_column_embeddings[:use_rows].to(
                device=embs.device,
                dtype=embs.dtype,
            )
        embs = self.feature_positional_embedding_embeddings(embs)
        grouped_features += embs[None, None]
        return grouped_features

    def _preprocess_and_embed_features(
        self,
        features: torch.Tensor,
        num_train_labels: int,
        batch_size: int,
    ) -> torch.Tensor:
        features = self.remove_constant_features_encoder(features)
        feature_groups = self.feature_group_reshape_encoder(features)
        num_feature_groups = self.feature_group_reshape_encoder.num_feature_groups
        nan_and_inf_indicator = self.nan_inf_indicator_encoder(feature_groups)
        feature_groups = self.nan_handling_encoder(
            feature_groups,
            single_eval_pos=num_train_labels,
        )
        feature_groups = self.input_normalization_encoder(
            feature_groups,
            single_eval_pos=num_train_labels,
        )
        feature_groups = self.variable_num_features_encoder(
            feature_groups,
            single_eval_pos=num_train_labels,
        )
        feature_groups = torch.cat([feature_groups, nan_and_inf_indicator], dim=-1)
        embedded_features = self.feature_group_embedder(feature_groups)
        embedded_features = embedded_features.unflatten(
            1, [batch_size, num_feature_groups]
        )
        embedded_features = embedded_features.transpose(0, 1)
        return self._add_column_embeddings(embedded_features)


class TabPFNYPreEncoder(nn.Module):
    """Preprocess and embed TabPFN targets."""

    def __init__(
        self,
        *,
        emsize: int,
        task_type: Literal["multiclass", "regression"],
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.input_size = int(emsize)
        self.task_type = task_type
        self.target_embedder = nn.Linear(
            ENCODING_SIZE_MULTIPLIER,
            self.input_size,
            device=device,
            dtype=dtype,
        )
        self.nan_inf_indicator_encoder = NanInfIndicatorEncoder(
            nan_indicator=NAN_INDICATOR,
            pos_inf_indicator=INFINITY_INDICATOR,
            neg_inf_indicator=NEG_INFINITY_INDICATOR,
        )

    def forward(
        self,
        y: torch.Tensor,
        *,
        num_train_rows: int,
        num_train_labels: int,
        batch_size: int,
    ) -> torch.Tensor:
        return self._preprocess_and_embed_targets(
            y=y,
            num_train_rows=num_train_rows,
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
        prepared_targets = _prepare_targets(
            y=y,
            num_train_rows=num_train_rows,
            batch_size=batch_size,
        )
        nan_and_inf_indicator = self.nan_inf_indicator_encoder(prepared_targets)
        prepared_targets = _impute_target_nan_and_inf(
            targets=prepared_targets,
            task_type=self.task_type,
            num_train_rows=num_train_labels,
        )
        prepared_targets = torch.cat([prepared_targets, nan_and_inf_indicator], dim=-1)
        embedded_targets = self.target_embedder(prepared_targets)
        return embedded_targets.transpose(0, 1)


class TabPFNPreEncoder(nn.Module):
    """Compatibility wrapper around separate X/Y pre-encoders."""

    def __init__(
        self,
        *,
        emsize: int,
        features_per_group: int,
        encoder_type: Literal["linear", "mlp"],
        encoder_mlp_hidden_dim: int,
        task_type: Literal["multiclass", "regression"],
        column_embeddings_path: str | Path,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.x_pre_encoder = TabPFNXPreEncoder(
            emsize=emsize,
            features_per_group=features_per_group,
            encoder_type=encoder_type,
            encoder_mlp_hidden_dim=encoder_mlp_hidden_dim,
            column_embeddings_path=column_embeddings_path,
            device=device,
            dtype=dtype,
        )
        self.y_pre_encoder = TabPFNYPreEncoder(
            emsize=emsize,
            task_type=task_type,
            device=device,
            dtype=dtype,
        )

    @property
    def feature_group_embedder(self) -> nn.Module:
        return self.x_pre_encoder.feature_group_embedder

    @property
    def target_embedder(self) -> nn.Module:
        return self.y_pre_encoder.target_embedder

    @property
    def pre_generated_column_embeddings(self) -> torch.Tensor:
        return self.x_pre_encoder.pre_generated_column_embeddings

    @pre_generated_column_embeddings.setter
    def pre_generated_column_embeddings(self, value: torch.Tensor) -> None:
        self.x_pre_encoder.pre_generated_column_embeddings = value

    @property
    def feature_positional_embedding_embeddings(self) -> nn.Module:
        return self.x_pre_encoder.feature_positional_embedding_embeddings

    @property
    def nan_handling_encoder(self) -> NanHandlingEncoder:
        return self.x_pre_encoder.nan_handling_encoder

    @property
    def input_normalization_encoder(self) -> InputNormalizationEncoder:
        return self.x_pre_encoder.input_normalization_encoder

    @property
    def variable_num_features_encoder(self) -> VariableNumFeaturesEncoder:
        return self.x_pre_encoder.variable_num_features_encoder

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

    def _add_column_embeddings(self, grouped_features: torch.Tensor) -> torch.Tensor:
        return self.x_pre_encoder._add_column_embeddings(grouped_features)

    def _preprocess_and_embed_features(
        self,
        features: torch.Tensor,
        num_train_labels: int,
        batch_size: int,
    ) -> torch.Tensor:
        del batch_size
        return self.x_pre_encoder._preprocess_and_embed_features(
            features=features,
            num_train_labels=num_train_labels,
            batch_size=features.shape[1],
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


def _torch_nanmean_include_inf(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    x_clean = x.clone()
    x_clean[torch.isinf(x_clean)] = torch.nan
    return torch.nanmean(x_clean, dim=dim)


def _remove_constant_features(features: torch.Tensor) -> torch.Tensor:
    return RemoveConstantFeaturesEncoder()(features)


def _pad_and_reshape_feature_groups(
    features: torch.Tensor,
    num_features_per_group: int,
) -> tuple[torch.Tensor, int]:
    encoder = FeatureGroupReshapeEncoder(
        num_features_per_group=num_features_per_group,
    )
    grouped_features = encoder(features)
    return grouped_features, encoder.num_feature_groups


def _impute_nan_and_inf_with_mean(
    x: torch.Tensor,
    num_train_rows: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    feature_means = _torch_nanmean_include_inf(x[:num_train_rows], dim=0)
    nan_mask = torch.logical_or(torch.isnan(x), torch.isinf(x))
    imputed = torch.where(nan_mask, feature_means.unsqueeze(0).expand_as(x), x)
    return imputed, nan_mask


def _impute_target_nan_and_inf(
    targets: torch.Tensor,
    task_type: Literal["multiclass", "regression"],
    num_train_rows: int,
) -> torch.Tensor:
    if task_type == "regression":
        return _impute_nan_and_inf_with_mean(targets, num_train_rows)[0]
    targets, nan_inf_mask = _impute_nan_and_inf_with_mean(
        targets,
        num_train_rows,
    )
    return torch.where(nan_inf_mask, targets.ceil(), targets)


def _generate_nan_and_inf_indicator(x: torch.Tensor) -> torch.Tensor:
    return NanInfIndicatorEncoder(
        nan_indicator=NAN_INDICATOR,
        pos_inf_indicator=INFINITY_INDICATOR,
        neg_inf_indicator=NEG_INFINITY_INDICATOR,
    )(x)


def _prepare_targets(
    y: torch.Tensor,
    num_train_rows: int,
    batch_size: int,
) -> torch.Tensor:
    num_train_labels = y.shape[0]
    if num_train_labels > num_train_rows:
        raise ValueError("No test rows provided.")
    targets = y.view(num_train_labels, 1 if y.ndim == 1 else batch_size, -1)
    return torch.nn.functional.pad(
        targets,
        (0, 0, 0, 0, 0, num_train_rows - num_train_labels),
        value=float("nan"),
    )
