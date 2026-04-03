from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

from rllm.types import ColType

from .pre_encoder import PreEncoder
from .col_encoder._embedding_encoder import EmbeddingEncoder
from .col_encoder._nan_handling_encoder import NanHandlingEncoder
from .col_encoder._remove_empty_features_encoder import RemoveEmptyFeaturesEncoder
from .col_encoder._variable_num_features_encoder import VariableNumFeaturesEncoder
from .col_encoder._input_normalization_encoder import InputNormalizationEncoder
from .col_encoder._frequency_feature_encoder import FrequencyFeatureEncoder
from .col_encoder._categorical_input_per_feature_encoder import (
    CategoricalInputPerFeatureEncoder,
)
from .col_encoder._input_normalization_encoder import torch_nanmean, torch_nanstd


class TabPFNPreEncoder(PreEncoder):
    r"""TabPFN-style pre-encoder built from sequential ColEncoder pipelines.

    This class maps TabPFN's encoder-step idea onto the project-wide
    ``Dict[ColType, List[ColEncoder]]`` interface.
    """

    def __init__(
        self,
        out_dim: int,
        metadata: Dict[ColType, List[Dict[str, Any]]],
        *,
        remove_empty_features: bool = True,
        remove_duplicate_features: bool = False,
        nan_handling_enabled: bool = True,
        normalize_on_train_only: bool = True,
        normalize_to_ranking: bool = False,
        normalize_x: bool = True,
        remove_outliers: bool = False,
        normalize_by_used_features: bool = True,
        encoder_use_bias: bool = False,
        num_frequencies: int = 0,
        normalize_by_sqrt: bool = True,
        fixed_num_features: Optional[int] = None,
        use_categorical_per_feature_encoder: bool = False,
        categorical_num_embs: int = 1000,
    ) -> None:
        if normalize_to_ranking:
            raise ValueError(
                "normalize_to_ranking=True is not supported in the current ColEncoder pipeline."
            )

        # Keep this flag for API parity with TabPFN v2 config.
        self.normalize_on_train_only = normalize_on_train_only
        self.nan_handling_enabled = nan_handling_enabled

        col_encoder_dict: Dict[ColType, list] = {}

        if ColType.NUMERICAL in metadata:
            num_chain = []

            if remove_empty_features:
                num_chain.append(RemoveEmptyFeaturesEncoder())

            # Keep the flag for API compatibility, but do not append this encoder:
            # TabPFNPreEncoder's sequential path does not execute duplicate-removal.
            del remove_duplicate_features

            if nan_handling_enabled:
                num_chain.append(NanHandlingEncoder())

            if normalize_x or remove_outliers:
                num_chain.append(
                    InputNormalizationEncoder(
                        normalize_x=normalize_x,
                        remove_outliers=remove_outliers,
                    )
                )

            num_features_target = fixed_num_features
            if num_features_target is None:
                num_features_target = len(metadata[ColType.NUMERICAL])
            num_chain.append(
                VariableNumFeaturesEncoder(
                    num_features=num_features_target,
                    normalize_by_used_features=normalize_by_used_features,
                    normalize_by_sqrt=normalize_by_sqrt,
                )
            )

            if num_frequencies > 0:
                num_chain.append(
                    FrequencyFeatureEncoder(num_frequencies=num_frequencies)
                )

            col_encoder_dict[ColType.NUMERICAL] = num_chain

        if ColType.CATEGORICAL in metadata:
            if use_categorical_per_feature_encoder:
                cat_chain = [
                    CategoricalInputPerFeatureEncoder(
                        out_dim=out_dim,
                        num_embs=categorical_num_embs,
                    )
                ]
            else:
                cat_chain = [EmbeddingEncoder(out_dim=out_dim)]
            col_encoder_dict[ColType.CATEGORICAL] = cat_chain

        if ColType.BINARY in metadata:
            # Binary columns are usually handled as low-cardinality categories.
            if use_categorical_per_feature_encoder:
                bin_chain = [
                    CategoricalInputPerFeatureEncoder(
                        out_dim=out_dim,
                        num_embs=max(categorical_num_embs, 8),
                    )
                ]
            else:
                bin_chain = [EmbeddingEncoder(out_dim=out_dim)]
            col_encoder_dict[ColType.BINARY] = bin_chain

        super().__init__(
            out_dim=out_dim, metadata=metadata, col_encoder_dict=col_encoder_dict
        )

        # Match TabPFN v2 checkpoint layout: a single ``nn.Linear`` as ``encoder.5.layer``
        # (see ``LinearInputEncoderStep`` in ``tabpfn_v2.encoders``). This replaces
        # ``LinearEncoder``'s per-column Parameters while keeping the same numerics.
        if ColType.NUMERICAL in metadata:
            num_features_target = fixed_num_features
            if num_features_target is None:
                num_features_target = len(metadata[ColType.NUMERICAL])
            linear_in_dim_local = 1
            if num_frequencies > 0:
                linear_in_dim_local = 1 + 2 * num_frequencies
            if nan_handling_enabled:
                linear_in_dim_local += 1
            self._linear_in_dim = linear_in_dim_local
            tabpfn_linear = torch.nn.Linear(
                num_features_target * linear_in_dim_local,
                out_dim,
                bias=encoder_use_bias,
            )
            linear_step = torch.nn.Module()
            linear_step.layer = tabpfn_linear
            self.add_module("5", linear_step)

    def forward(
        self,
        feat_dict: Dict[ColType, torch.Tensor] | Dict[str, torch.Tensor],
        return_dict: bool = False,
        **kwargs: object,
    ) -> torch.Tensor | Dict[ColType, torch.Tensor]:
        if "main" not in feat_dict:
            return super().forward(feat_dict, return_dict=return_dict)

        if return_dict:
            raise ValueError(
                "TabPFNPreEncoder does not support return_dict=True for 'main' inputs."
            )

        x = feat_dict["main"]
        if x.ndim != 3:
            raise ValueError(
                f"Expected TabPFN input with shape [S, BG, F], got {tuple(x.shape)}."
            )

        seq_len, batch_groups, _num_features = x.shape
        numerical_key = ColType.NUMERICAL.value
        if numerical_key not in self.col_encoder_dict:
            raise ValueError("TabPFNPreEncoder requires a numerical encoder chain.")

        single_eval_pos = kwargs.get("single_eval_pos")
        if single_eval_pos is None:
            single_eval_pos = x.shape[0]
        if single_eval_pos <= 0:
            raise ValueError("single_eval_pos must be positive for TabPFNPreEncoder.")

        x_work = x
        col_encoders = list(self.col_encoder_dict[numerical_key])

        for col_encoder in col_encoders:
            if isinstance(
                col_encoder,
                (
                    RemoveEmptyFeaturesEncoder,
                    NanHandlingEncoder,
                    InputNormalizationEncoder,
                    VariableNumFeaturesEncoder,
                ),
            ):
                x_work = col_encoder(
                    x_work,
                    single_eval_pos=single_eval_pos,
                    normalize_on_train_only=self.normalize_on_train_only,
                )
                continue

            if isinstance(col_encoder, FrequencyFeatureEncoder):
                x_flat = x_work.reshape(-1, x_work.shape[-1])
                x_flat = col_encoder(x_flat)
                x_work = x_flat.reshape(
                    seq_len,
                    x_work.shape[1],
                    x_work.shape[2],
                    -1,
                )
                continue

            x_flat = x_work.reshape(-1, x_work.shape[-1])
            x_work = col_encoder(x_flat).reshape(seq_len, batch_groups, -1)

        lin = self.get_submodule("5").layer
        if x_work.ndim == 3:
            x_flat = x_work.reshape(-1, x_work.shape[-1], 1)
        else:
            x_flat = x_work.reshape(-1, x_work.shape[2], x_work.shape[3])

        if self.nan_handling_enabled:
            nan_indicator = torch.logical_or(torch.isnan(x), torch.isinf(x)).to(
                x_flat.dtype
            )
            nan_flat = nan_indicator.reshape(-1, nan_indicator.shape[-1], 1)
            x_flat = torch.cat([x_flat, nan_flat], dim=-1)
        x_stats_source = (
            x_work[:single_eval_pos]
            if self.normalize_on_train_only and single_eval_pos is not None
            else x_work
        )
        if x_stats_source.ndim == 3:
            x_stats_flat = x_stats_source.reshape(-1, x_stats_source.shape[-1], 1)
        else:
            x_stats_flat = x_stats_source.reshape(
                -1, x_stats_source.shape[2], x_stats_source.shape[3]
            )
        if self.nan_handling_enabled:
            x_indicator_source = (
                x[:single_eval_pos]
                if self.normalize_on_train_only and single_eval_pos is not None
                else x
            )
            nan_stats = torch.logical_or(
                torch.isnan(x_indicator_source),
                torch.isinf(x_indicator_source),
            ).to(x_stats_flat.dtype)
            nan_stats = nan_stats.reshape(-1, nan_stats.shape[-1], 1)
            x_stats_flat = torch.cat([x_stats_flat, nan_stats], dim=-1)
        mean = torch_nanmean(x_stats_flat, axis=0).unsqueeze(0)  # type: ignore
        std = torch_nanstd(x_stats_flat, axis=0).unsqueeze(0)
        std = torch.where(std == 0, torch.ones_like(std), std)
        x_norm = (x_flat - mean) / (std + 1e-16)
        x_in = x_norm.reshape(x_norm.shape[0], -1).to(lin.weight.dtype)
        out = lin(x_in)
        return out.reshape(seq_len, batch_groups, -1)
