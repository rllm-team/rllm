#  Copyright (c) Prior Labs GmbH 2025.

from __future__ import annotations

import logging
import math
import re
import urllib.request
import urllib.response
import warnings
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Literal
from urllib.error import URLError

import torch
from torch import nn

from rllm.nn.encoder.tabpfn_pre_encoder import TabPFNPreEncoder
from rllm.nn.encoder.tabpfn_y_pre_encoder import TabPFNYPreEncoder
from rllm.types import ColType, StatType

from .bar_distribution import FullSupportBarDistribution
from .config import InferenceConfig
from .tabpfn_backbone import PerFeatureTransformer

logger = logging.getLogger(__name__)


def get_loss_criterion(
    config: InferenceConfig,
) -> nn.BCEWithLogitsLoss | nn.CrossEntropyLoss | FullSupportBarDistribution:
    # NOTE: We don't seem to have any of these
    if config.max_num_classes == 2:
        return nn.BCEWithLogitsLoss(reduction="none")

    if config.max_num_classes > 2:
        return nn.CrossEntropyLoss(reduction="none")

    assert config.max_num_classes == 0
    num_buckets = config.num_buckets

    # NOTE: This just seems to get overriddden in the module loading from `state_dict`
    # dummy values, extra bad s.t. one realizes if they are used for training
    borders = torch.arange(num_buckets + 1).float() * 10_000
    borders = borders * 3  # Used to be `config.get("bucket_scaling", 3)`

    return FullSupportBarDistribution(borders, ignore_nan_targets=True)


def _preprocess_config(config: dict) -> InferenceConfig:
    config["task_type"]
    batch_size = config["batch_size"]
    agg_k_grads = config.get("aggregate_k_gradients")

    if agg_k_grads is None:
        if not math.log(batch_size, 2).is_integer():
            raise ValueError(f"batch_size must be pow of 2, got {config['batch_size']}")

        second_dim_tokens = config.get("num_global_att_tokens ", config["seq_len"])
        memory_factor = (
            batch_size
            * config["nlayers"]
            * config["emsize"]
            * config["seq_len"]
            * second_dim_tokens
        )
        standard_memory_factor = 16 * 12 * 512 * 1200 * 1200
        agg_k_grads = math.ceil(memory_factor / (standard_memory_factor * 1.1))
        config["aggregate_k_gradients"] = agg_k_grads

        # Make sure that batch size is power of two
        config["batch_size"] = int(
            math.pow(2, math.floor(math.log(batch_size / agg_k_grads, 2))),
        )
        config["num_steps"] = math.ceil(config["num_steps"] * agg_k_grads)

        # Make sure that batch_size_per_gp_sample is power of two
        assert math.log(config["batch_size_per_gp_sample"], 2) % 1 == 0

    config.setdefault("recompute_attn", False)
    return InferenceConfig.from_dict(config)


def get_encoder(  # noqa: PLR0913
    *,
    num_features: int,
    embedding_size: int,
    remove_empty_features: bool,
    remove_duplicate_features: bool,
    nan_handling_enabled: bool,
    normalize_on_train_only: bool,
    normalize_to_ranking: bool,
    normalize_x: bool,
    remove_outliers: bool,
    normalize_by_used_features: bool,
    encoder_use_bias: bool,
) -> nn.Module:
    # Build minimal numerical metadata required by ColEncoder-based TabPFNPreEncoder.
    # Statistics are neutral defaults because this TabPFN loading path does not provide
    # dataset-level metadata at checkpoint load time.
    numerical_stats = [
        {
            StatType.MEAN: 0.0,
            StatType.STD: 1.0,
            StatType.MIN: -1e9,
            StatType.MAX: 1e9,
        }
        for _ in range(num_features)
    ]
    metadata = {ColType.NUMERICAL: numerical_stats}

    return TabPFNPreEncoder(
        out_dim=embedding_size,
        metadata=metadata,
        remove_empty_features=remove_empty_features,
        remove_duplicate_features=remove_duplicate_features,
        nan_handling_enabled=nan_handling_enabled,
        normalize_on_train_only=normalize_on_train_only,
        normalize_to_ranking=normalize_to_ranking,
        normalize_x=normalize_x,
        remove_outliers=remove_outliers,
        normalize_by_used_features=normalize_by_used_features,
        encoder_use_bias=encoder_use_bias,
        fixed_num_features=num_features,
    )


def get_y_encoder(
    *,
    num_inputs: int,
    embedding_size: int,
    nan_handling_y_encoder: bool,
    max_num_classes: int,
) -> nn.Module:
    return TabPFNYPreEncoder(
        num_inputs=num_inputs,
        embedding_size=embedding_size,
        nan_handling_y_encoder=nan_handling_y_encoder,
        max_num_classes=max_num_classes,
    )


def _find_last_matching_key(
    state_dict: dict[str, torch.Tensor],
    pattern: str,
) -> str | None:
    matches: list[tuple[int, str]] = []
    regex = re.compile(pattern)
    for key in state_dict:
        match = regex.fullmatch(key)
        if match is not None:
            matches.append((int(match.group(1)), key))
    if not matches:
        return None
    matches.sort()
    return matches[-1][1]


def _convert_old_linear_weight(
    tensor: torch.Tensor,
    target_shape: torch.Size,
) -> torch.Tensor:
    if tensor.shape == target_shape:
        return tensor

    if tensor.ndim == 2 and len(target_shape) == 3:
        num_cols, in_dim, out_dim = target_shape
        if in_dim == 1 and tensor.shape[0] == out_dim and tensor.shape[1] >= num_cols:
            converted = tensor[:, :num_cols].transpose(0, 1).unsqueeze(1)
            if converted.shape == target_shape:
                return converted

    return tensor


def _convert_old_linear_bias(
    tensor: torch.Tensor,
    target_shape: torch.Size,
) -> torch.Tensor:
    if tensor.shape == target_shape:
        return tensor

    if tensor.ndim == 1 and len(target_shape) == 2:
        converted = tensor.unsqueeze(0).expand(target_shape[0], -1) / target_shape[0]
        if converted.shape == target_shape:
            return converted

    return tensor


def _remap_legacy_tabpfn_state_dict(
    state_dict: dict[str, torch.Tensor],
    model: nn.Module,
) -> dict[str, torch.Tensor]:
    remapped_state = model.state_dict()

    for key, value in state_dict.items():
        if key in remapped_state and remapped_state[key].shape == value.shape:
            remapped_state[key] = value

    encoder_linear_weight_key = next(
        key
        for key in remapped_state
        if key.startswith("encoder.col_encoder_dict.numerical.")
        and key.endswith(".weight")
    )
    encoder_linear_bias_key = encoder_linear_weight_key.replace(".weight", ".bias")
    y_weight_key = "y_encoder.proj.weight"
    y_bias_key = "y_encoder.proj.bias"

    source_encoder_weight = _find_last_matching_key(
        state_dict,
        r"encoder\.(\d+)\.layer\.weight",
    )
    source_encoder_bias = _find_last_matching_key(
        state_dict,
        r"encoder\.(\d+)\.layer\.bias",
    )
    source_y_weight = _find_last_matching_key(
        state_dict,
        r"y_encoder\.(\d+)\.layer\.weight",
    )
    source_y_bias = _find_last_matching_key(
        state_dict,
        r"y_encoder\.(\d+)\.layer\.bias",
    )

    if source_encoder_weight is not None:
        remapped_state[encoder_linear_weight_key] = _convert_old_linear_weight(
            state_dict[source_encoder_weight],
            remapped_state[encoder_linear_weight_key].shape,
        )

    if source_encoder_bias is not None:
        remapped_state[encoder_linear_bias_key] = _convert_old_linear_bias(
            state_dict[source_encoder_bias],
            remapped_state[encoder_linear_bias_key].shape,
        )

    if source_y_weight is not None:
        remapped_state[y_weight_key] = state_dict[source_y_weight]

    if source_y_bias is not None:
        remapped_state[y_bias_key] = state_dict[source_y_bias]

    return remapped_state


def load_model(
    *,
    path: Path,
    model_seed: int,
) -> tuple[
    PerFeatureTransformer,
    InferenceConfig,
    object,
]:
    """Loads a model from a given path.

    Args:
        path: Path to the checkpoint
        model_seed: The seed to use for the model

    Returns:
        Tuple of (model, config, loss_criterion). loss_criterion is a
        FullSupportBarDistribution for regression models, or None for classifiers.
    """
    # Catch the `FutureWarning` that torch raises. This should be dealt with!
    # The warning is raised due to `torch.load`, which advises against ckpt
    # files that contain non-tensor data.
    # This `weightes_only=None` is the default value. In the future this will default to
    # `True`, dissallowing loading of arbitrary objects.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        checkpoint = torch.load(path, map_location="cpu", weights_only=None)

    state_dict = checkpoint["state_dict"]
    raw_config: dict[str, Any] = checkpoint["config"]
    config = _preprocess_config(raw_config)

    criterion_state_keys = [k for k in state_dict if "criterion." in k]
    loss_criterion = get_loss_criterion(config)
    if isinstance(loss_criterion, FullSupportBarDistribution):
        # Remove from state dict
        criterion_state = {
            k.replace("criterion.", ""): state_dict.pop(k) for k in criterion_state_keys
        }
        loss_criterion.load_state_dict(criterion_state)
    else:
        assert len(criterion_state_keys) == 0, criterion_state_keys

    n_out: int
    if config.max_num_classes == 2:
        n_out = 1
    elif config.max_num_classes > 2:
        n_out = config.max_num_classes
    elif config.max_num_classes == 0:
        n_out = loss_criterion.num_bars

    model = PerFeatureTransformer(
        seed=model_seed,
        # Things that were explicitly passed inside `build_model()`
        encoder=get_encoder(
            num_features=config.features_per_group,
            embedding_size=config.emsize,
            remove_empty_features=config.remove_empty_features,
            remove_duplicate_features=config.remove_duplicate_features,
            nan_handling_enabled=config.nan_handling_enabled,
            normalize_on_train_only=config.normalize_on_train_only,
            normalize_to_ranking=config.normalize_to_ranking,
            normalize_x=config.normalize_x,
            remove_outliers=config.remove_outliers,
            normalize_by_used_features=config.normalize_by_used_features,
            encoder_use_bias=config.encoder_use_bias,
        ),
        y_encoder=get_y_encoder(
            num_inputs=1,
            embedding_size=config.emsize,
            nan_handling_y_encoder=config.nan_handling_y_encoder,
            max_num_classes=config.max_num_classes,
        ),
        nhead=config.nhead,
        ninp=config.emsize,
        nhid=config.emsize * config.nhid_factor,
        nlayers=config.nlayers,
        features_per_group=config.features_per_group,
        cache_trainset_representation=True,
        #
        # Based on not being present in config or otherwise, these were default values
        init_method=None,
        decoder_dict={"standard": (None, n_out)},
        use_encoder_compression_layer=False,
        #
        # These were extra things passed in through `**model_extra_args`
        # or `**extra_model_kwargs` and were present in the config
        recompute_attn=config.recompute_attn,
        recompute_layer=config.recompute_layer,
        feature_positional_embedding=config.feature_positional_embedding,
        use_separate_decoder=config.use_separate_decoder,
        #
        # These are things that had default values from config.get() but were not
        # present in any config.
        layer_norm_with_elementwise_affine=False,
        nlayers_decoder=None,
        pre_norm=False,
        #
        # These seem to map to `**layer_config` in the init of `PerFeatureTransformer`
        # Which got passed to the `PerFeatureEncoderLayer(**layer_config)`
        multiquery_item_attention=config.multiquery_item_attention,  # False
        multiquery_item_attention_for_test_set=config.multiquery_item_attention_for_test_set,  # True  # noqa: E501
        # Is either 1.0 or None in the configs, which lead to the default of 1.0 anywho
        attention_init_gain=(
            config.attention_init_gain
            if config.attention_init_gain is not None
            else 1.0
        ),
        # Is True, False in the config or not present,
        # with the default of the `PerFeatureEncoderLayer` being False,
        # which is what the value would have mapped to if the config had not present
        two_sets_of_queries=(
            config.two_sets_of_queries
            if config.two_sets_of_queries is not None
            else False
        ),
    )
    remapped_state_dict = _remap_legacy_tabpfn_state_dict(state_dict, model)
    model.load_state_dict(remapped_state_dict, strict=True)
    model.eval()
    criterion_to_return = (
        loss_criterion
        if isinstance(loss_criterion, FullSupportBarDistribution)
        else None
    )
    return model, config, criterion_to_return
