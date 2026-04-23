#  Copyright (c) Prior Labs GmbH 2025.

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn

from rllm.nn.loss import FullSupportBarDistribution
from rllm.types import ColType

from .config import TabPFNVersionConfig, build_version_config
from .tabpfn_backbone import TabPFNConfig
from .tabpfn_model import TabPFNModel
from .tabpfn_utils import PREGENERATED_COLUMN_EMBEDDINGS_FILENAME

logger = logging.getLogger(__name__)


def _infer_model_version_from_path(path: Path) -> Literal["v2_6"]:
    lowered_name = path.name.lower()
    if "v2.6" in lowered_name or "v2_6" in lowered_name:
        return "v2_6"
    raise RuntimeError(
        "Only the retained TabPFN checkpoint format is supported. "
        f"Expected a matching filename, got: {path.name}"
    )


def get_loss_criterion(
    *,
    model_type: Literal["clf", "reg"],
    num_buckets: int,
) -> nn.CrossEntropyLoss | FullSupportBarDistribution:
    if model_type == "clf":
        return nn.CrossEntropyLoss(reduction="none")

    borders = torch.arange(num_buckets + 1).float() * 10_000 * 3
    return FullSupportBarDistribution(borders, ignore_nan_targets=True)


def load_model(
    *,
    path: Path,
    model_seed: int,
    model_type: Literal["clf", "reg"],
    metadata: dict[ColType, list[dict[Any, Any]]] | None = None,
    version_config: TabPFNVersionConfig | None = None,
    strict_version_match: bool = True,
) -> tuple[
    nn.Module,
    dict[str, Any],
    object,
]:
    """Load the retained TabPFN checkpoint."""
    del model_seed, metadata

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        checkpoint = torch.load(path, map_location="cpu", weights_only=None)
    state_dict = checkpoint["state_dict"]
    config = checkpoint.get("config")

    inferred_model_version = _infer_model_version_from_path(path)
    if version_config is None:
        version_config = build_version_config(
            model_type,
            model_version=inferred_model_version,
        )
    elif version_config.model_version != inferred_model_version:
        message = (
            "Checkpoint/runtime version mismatch: "
            f"requested={version_config.model_version}, "
            f"inferred={inferred_model_version}, "
            f"path={path}. "
            "Use the checkpoint that matches the requested version."
        )
        if strict_version_match:
            raise RuntimeError(message)
        warnings.warn(
            f"{message} Falling back to checkpoint-inferred runtime settings.",
            stacklevel=2,
        )
        version_config = build_version_config(
            model_type,
            model_version=inferred_model_version,
        )

    if model_type == "reg":
        config["max_num_classes"] = 0
        config["task_type"] = "regression"

    criterion_state_keys = [k for k in state_dict if "criterion." in k]
    loss_criterion = get_loss_criterion(
        model_type=model_type,
        num_buckets=int(config["num_buckets"]),
    )
    if isinstance(loss_criterion, FullSupportBarDistribution):
        criterion_state = {
            k.replace("criterion.", ""): state_dict.pop(k) for k in criterion_state_keys
        }
        loss_criterion.load_state_dict(criterion_state)
    else:
        assert len(criterion_state_keys) == 0, criterion_state_keys

    if config["max_num_classes"] == 2:
        n_out = 1
    elif config["max_num_classes"] > 2:
        n_out = int(config["max_num_classes"])
    else:
        n_out = int(loss_criterion.num_bars)

    task_type: Literal["multiclass", "regression"] = (
        "multiclass" if int(config["max_num_classes"]) > 0 else "regression"
    )
    model_config_kwargs = dict(
        emsize=int(config["emsize"]),
        nlayers=version_config.resolve_nlayers(model_type, int(config["nlayers"])),
        nhead=int(config["nhead"]),
        features_per_group=version_config.resolve_feature_group_size(
            int(config["features_per_group"])
        ),
        num_thinking_rows=version_config.n_thinking_rows,
        encoder_type=(
            "mlp"
            if task_type == "regression" and version_config.use_regression_mlp_encoder
            else "linear"
        ),
        encoder_mlp_hidden_dim=version_config.regression_mlp_hidden_dim,
    )
    pregenerated_path = path.parent / PREGENERATED_COLUMN_EMBEDDINGS_FILENAME
    if not pregenerated_path.exists():
        raise FileNotFoundError(
            f"Required companion file '{pregenerated_path.name}' was not found "
            f"next to checkpoint '{path.name}'."
        )

    model = TabPFNModel(
        config=TabPFNConfig(**model_config_kwargs),
        n_out=n_out,
        task_type=task_type,
        column_embeddings_path=pregenerated_path,
    )

    load_checkpoint_compatible(
        model=model,
        state_dict=state_dict,
        source_checkpoint_version=inferred_model_version,
        source_model_version=inferred_model_version,
        target_model_version=version_config.model_version,
    )

    model.eval()
    criterion_to_return = (
        loss_criterion
        if isinstance(loss_criterion, FullSupportBarDistribution)
        else None
    )
    return model, config, criterion_to_return


def load_checkpoint_compatible(
    *,
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
    source_checkpoint_version: str,
    source_model_version: str,
    target_model_version: str,
) -> None:
    """Load a retained checkpoint into the retained runtime."""
    state_dict = _remap_checkpoint_keys(state_dict)
    target_state = model.state_dict()

    unexpected_keys = sorted(k for k in state_dict if k not in target_state)
    missing_keys = sorted(k for k in target_state if k not in state_dict)
    shape_mismatch_keys = sorted(
        k
        for k in state_dict
        if k in target_state and state_dict[k].shape != target_state[k].shape
    )
    if missing_keys or unexpected_keys or shape_mismatch_keys:
        msg_lines = [
            "Strict checkpoint loading failed.",
            f"source_ckpt={source_checkpoint_version}",
            f"source_model={source_model_version}",
            f"target_model={target_model_version}",
        ]
        if missing_keys:
            msg_lines.append(
                f"missing_keys={missing_keys[:8]}"
                + (" ..." if len(missing_keys) > 8 else "")
            )
        if unexpected_keys:
            msg_lines.append(
                f"unexpected_keys={unexpected_keys[:8]}"
                + (" ..." if len(unexpected_keys) > 8 else "")
            )
        if shape_mismatch_keys:
            msg_lines.append(
                f"shape_mismatch_keys={shape_mismatch_keys[:8]}"
                + (" ..." if len(shape_mismatch_keys) > 8 else "")
            )
        raise RuntimeError("\n".join(msg_lines))
    model.load_state_dict(state_dict, strict=True)
    logger.info(
        "Loaded checkpoint strictly: source_ckpt=%s source_model=%s target_model=%s",
        source_checkpoint_version,
        source_model_version,
        target_model_version,
    )


def _remap_checkpoint_keys(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Map legacy flat checkpoint keys onto the new model/backbone split."""

    remapped: dict[str, torch.Tensor] = {}

    for key, value in state_dict.items():
        new_key = key
        if key.startswith("blocks."):
            new_key = f"backbone.{key}"
        elif key.startswith("feature_group_embedder."):
            new_key = f"x_pre_encoder.{key}"
        elif key.startswith("target_embedder."):
            new_key = f"y_pre_encoder.{key}"
        elif key.startswith("feature_positional_embedding_embeddings."):
            new_key = f"x_pre_encoder.{key}"
        elif key.startswith("pre_generated_column_embeddings"):
            new_key = f"x_pre_encoder.{key}"
        remapped[new_key] = value

    return remapped
