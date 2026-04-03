#  Copyright (c) Prior Labs GmbH 2025.

from __future__ import annotations

import logging
import re
import warnings
from pathlib import Path
from typing import Any, Callable, Literal

import torch
from torch import nn

from rllm.nn.loss import FullSupportBarDistribution
from rllm.types import ColType

from .tabpfn_backbone import PerFeatureTransformer

logger = logging.getLogger(__name__)


def get_loss_criterion(
    *,
    model_type: Literal["clf", "reg"],
    num_buckets: int,
) -> nn.BCEWithLogitsLoss | nn.CrossEntropyLoss | FullSupportBarDistribution:
    if model_type == "clf":
        return nn.CrossEntropyLoss(reduction="none")

    # model_type == "reg"
    borders = torch.arange(num_buckets + 1).float() * 10_000 * 3
    return FullSupportBarDistribution(borders, ignore_nan_targets=True)


def _find_last_matching_key(
    state_dict: dict[str, torch.Tensor], pattern: str
) -> str | None:
    """Return the last checkpoint key matching pattern with integer group(1)."""
    regex = re.compile(pattern)
    best: tuple[int, str] | None = None
    for key in state_dict:
        m = regex.fullmatch(key)
        if m is None:
            continue
        idx = int(m.group(1))
        if best is None or idx > best[0]:
            best = (idx, key)
    return None if best is None else best[1]


def _pick_target_weight_bias_keys(
    remapped_state: dict[str, torch.Tensor],
    *,
    weight_candidates: tuple[str, ...],
    fallback_weight_predicate: Callable[[str], bool] | None = None,
) -> tuple[str, str | None]:
    weight_key = next((k for k in weight_candidates if k in remapped_state), None)
    if weight_key is None and fallback_weight_predicate is not None:
        weight_key = next((k for k in remapped_state if fallback_weight_predicate(k)), None)
    if weight_key is None:
        raise KeyError("Could not find a target weight key in remapped_state.")
    bias_key = weight_key.replace(".weight", ".bias")
    if bias_key not in remapped_state:
        bias_key = None
    return weight_key, bias_key


def _adapt_encoder_weight(
    src: torch.Tensor, tgt_shape: torch.Size
) -> torch.Tensor | None:
    if src.shape == tgt_shape:
        return src
    if src.ndim == 2 and len(tgt_shape) == 3:
        num_cols, in_dim, out_dim = tgt_shape
        if in_dim == 1 and src.shape[0] == out_dim and src.shape[1] >= num_cols:
            return src[:, :num_cols].transpose(0, 1).unsqueeze(1)
    return None


def _adapt_encoder_bias(src: torch.Tensor, tgt_shape: torch.Size) -> torch.Tensor | None:
    if src.shape == tgt_shape:
        return src
    if src.ndim == 1 and len(tgt_shape) == 2:
        return src.unsqueeze(0).expand(tgt_shape[0], -1) / tgt_shape[0]
    return None

def _remap_legacy_tabpfn_state_dict(
    state_dict: dict[str, torch.Tensor],
    model: nn.Module,
) -> dict[str, torch.Tensor]:
    remapped_state = model.state_dict()

    for key, value in state_dict.items():
        if key in remapped_state and remapped_state[key].shape == value.shape:
            remapped_state[key] = value

    encoder_linear_weight_key, encoder_linear_bias_key = _pick_target_weight_bias_keys(
        remapped_state,
        weight_candidates=("encoder.5.layer.weight",),
        fallback_weight_predicate=lambda k: (
            k.startswith("encoder.col_encoder_dict.numerical.") and k.endswith(".weight")
        ),
    )
    y_weight_key, y_bias_key = _pick_target_weight_bias_keys(
        remapped_state,
        weight_candidates=("y_encoder.2.layer.weight", "y_encoder.proj.weight"),
    )

    source_encoder_weight = _find_last_matching_key(
        state_dict, r"encoder\.(\d+)\.layer\.weight"
    )
    source_encoder_bias = _find_last_matching_key(
        state_dict, r"encoder\.(\d+)\.layer\.bias"
    )
    source_y_weight = _find_last_matching_key(
        state_dict, r"y_encoder\.(\d+)\.layer\.weight"
    )
    source_y_bias = _find_last_matching_key(
        state_dict, r"y_encoder\.(\d+)\.layer\.bias"
    )

    if source_encoder_weight is not None:
        src = state_dict[source_encoder_weight]
        adapted = _adapt_encoder_weight(src, remapped_state[encoder_linear_weight_key].shape)
        if adapted is not None:
            remapped_state[encoder_linear_weight_key] = adapted

    if source_encoder_bias is not None and encoder_linear_bias_key is not None:
        src = state_dict[source_encoder_bias]
        adapted = _adapt_encoder_bias(src, remapped_state[encoder_linear_bias_key].shape)
        if adapted is not None:
            remapped_state[encoder_linear_bias_key] = adapted

    if source_y_weight is not None:
        src = state_dict[source_y_weight]
        if src.shape == remapped_state[y_weight_key].shape:
            remapped_state[y_weight_key] = src

    if source_y_bias is not None and y_bias_key is not None:
        src = state_dict[source_y_bias]
        if src.shape == remapped_state[y_bias_key].shape:
            remapped_state[y_bias_key] = src

    return remapped_state


def load_model(
    *,
    path: Path,
    model_seed: int,
    model_type: Literal["clf", "reg"],
    metadata: dict[ColType, list[dict[Any, Any]]] | None = None,
) -> tuple[
    PerFeatureTransformer,
    dict[str, Any],
    object,
]:
    """Loads a model from a given path.

    Args:
        path: Path to the checkpoint
        model_seed: The seed to use for the model
        metadata: Optional ``TabPFNPreEncoder`` column metadata (e.g. per-column stats).
            If omitted, neutral numerical placeholders are used (same as before).

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
    config = checkpoint.get("config")
    if model_type == "reg":
        config["max_num_classes"] = 0
        config["task_type"] = "regression"

    emsize = int(config["emsize"])
    nhid_factor = int(config["nhid_factor"])
    nlayers = int(config["nlayers"])

    criterion_state_keys = [k for k in state_dict if "criterion." in k]
    loss_criterion = get_loss_criterion(
        model_type=model_type,
        num_buckets=int(config["num_buckets"]),
    )
    if isinstance(loss_criterion, FullSupportBarDistribution):
        # Remove from state dict
        criterion_state = {
            k.replace("criterion.", ""): state_dict.pop(k) for k in criterion_state_keys
        }
        loss_criterion.load_state_dict(criterion_state)
    else:
        assert len(criterion_state_keys) == 0, criterion_state_keys

    n_out: int
    if config["max_num_classes"] == 2:
        n_out = 1
    elif config["max_num_classes"] > 2:
        n_out = config["max_num_classes"]
    elif config["max_num_classes"] == 0:
        n_out = loss_criterion.num_bars

    model = PerFeatureTransformer(
        seed=model_seed,
        encoder_num_features=config["features_per_group"],
        remove_empty_features=True,
        remove_duplicate_features=config["remove_duplicate_features"],
        nan_handling_enabled=True,
        normalize_on_train_only=True,
        normalize_to_ranking=False,
        normalize_x=True,
        remove_outliers=False,
        normalize_by_used_features=True,
        encoder_use_bias=False,
        encoder_metadata=metadata,
        y_num_inputs=1,
        nan_handling_y_encoder=True,
        max_num_classes=config["max_num_classes"],
        nhead=config["nhead"],
        ninp=emsize,
        nhid=emsize * nhid_factor,
        nlayers=nlayers,
        features_per_group=config["features_per_group"],
        cache_trainset_representation=True,
        #
        # Based on not being present in config or otherwise, these were default values
        init_method=None,
        decoder_dict={"standard": (None, n_out)},
        #
        # These were extra things passed in through `**model_extra_args`
        # or `**extra_model_kwargs` and were present in the config
        recompute_attn=False,
        recompute_layer=True,
        feature_positional_embedding="subspace",
        #
        # These are things that had default values from config.get() but were not
        # present in any config.
        layer_norm_with_elementwise_affine=False,
        pre_norm=False,
        #
        # These seem to map to `**layer_config` in the init of `PerFeatureTransformer`
        # Which got passed to the `PerFeatureEncoderLayer(**layer_config)`
        multiquery_item_attention=False,
        multiquery_item_attention_for_test_set=True,
        attention_init_gain=1.0,
        two_sets_of_queries=False,
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
