#  Copyright (c) Prior Labs GmbH 2025.

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn

from rllm.nn.loss import FullSupportBarDistribution
from rllm.utils import download_model_from_huggingface, download_url

from .backbone import TabPFNBackbone
from .utils import PREGENERATED_COLUMN_EMBEDDINGS_FILENAME

def _load_criterion_state(
    *,
    state_dict: dict[str, torch.Tensor],
    model_type: Literal["clf", "reg"],
    num_buckets: int,
) -> tuple[FullSupportBarDistribution | nn.CrossEntropyLoss, dict[str, torch.Tensor]]:
    criterion_state_keys = [k for k in state_dict if "criterion." in k]
    loss_criterion = get_loss_criterion(
        model_type=model_type,
        num_buckets=num_buckets,
    )
    if not isinstance(loss_criterion, FullSupportBarDistribution):
        assert len(criterion_state_keys) == 0, criterion_state_keys
        return loss_criterion, state_dict

    criterion_state = {
        k.replace("criterion.", ""): state_dict.pop(k) for k in criterion_state_keys
    }
    loss_criterion.load_state_dict(criterion_state)
    return loss_criterion, state_dict


def _build_model_config_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "emsize": int(config["emsize"]),
        "nlayers": int(config["nlayers"]),
        "nhead": int(config["nhead"]),
        "features_per_group": int(config["features_per_group"]),
        "num_thinking_rows": int(config["num_thinking_rows"]),
        "encoder_type": config["encoder_type"],
        "encoder_mlp_hidden_dim": int(config["encoder_mlp_hidden_dim"]),
    }


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
    model_type: Literal["clf", "reg"],
) -> tuple[
    nn.Module,
    dict[str, Any],
    object,
]:
    """Load a retained TabPFN v2.6 checkpoint into the local runtime."""

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state_dict = checkpoint["state_dict"]
    config = checkpoint.get("config")
    if config is None:
        raise ValueError("Config not found in checkpoint")
    if model_type == "reg":
        config["max_num_classes"] = 0
        config["task_type"] = "regression"
    loss_criterion, state_dict = _load_criterion_state(
        state_dict=state_dict,
        model_type=model_type,
        num_buckets=int(config["num_buckets"]),
    )

    max_num_classes = int(config["max_num_classes"])
    if max_num_classes == 2:
        n_out = 1
    elif max_num_classes > 2:
        n_out = max_num_classes
    else:
        n_out = int(loss_criterion.num_bars)

    task_type: Literal["multiclass", "regression"] = (
        "multiclass" if max_num_classes > 0 else "regression"
    )
    model_config_kwargs = _build_model_config_kwargs(config)
    pregenerated_path = path.parent / PREGENERATED_COLUMN_EMBEDDINGS_FILENAME
    if not pregenerated_path.exists():
        raise FileNotFoundError(
            f"Required companion file '{pregenerated_path.name}' was not found "
            f"next to checkpoint '{path.name}'."
        )

    model = TabPFNBackbone(
        **model_config_kwargs,
        n_out=n_out,
        task_type=task_type,
        column_embeddings_path=pregenerated_path,
    )

    model.load_state_dict(_remap_checkpoint_keys(state_dict), strict=True)
    model.eval()
    criterion_to_return = (
        loss_criterion
        if isinstance(loss_criterion, FullSupportBarDistribution)
        else None
    )
    return model, config, criterion_to_return


def initialize_tabpfn_model(
    model_dir: str,
    model_type: str,
) -> tuple[torch.nn.Module, dict[str, Any], object]:
    """Load the TabPFN model, config, and regression criterion when applicable."""
    # Determine filename (hardcoded to v2.6 retained checkpoints)
    model_name = (
        "tabpfn-v2.6-classifier-v2.6_default.ckpt"
        if model_type == "clf"
        else "tabpfn-v2.6-regressor-v2.6_default.ckpt"
    )
    model_path = os.path.join(model_dir, model_name)

    print(f"Loading model from {model_dir}...")
    if not os.path.exists(model_path):
        success = download_model_from_huggingface(
            repo="Prior-Labs/tabpfn_2_6",
            model_name=model_name,
            download_path=model_dir,
        )
        if not success:
            raise Exception(f"Failed to download model from Hugging Face: {model_name}")

    column_embedding_path = os.path.join(
        model_dir,
        PREGENERATED_COLUMN_EMBEDDINGS_FILENAME,
    )
    if not os.path.exists(column_embedding_path):
        url = (
            "https://raw.githubusercontent.com/PriorLabs/TabPFN/main/"
            "src/tabpfn/architectures/shared/tabpfn_col_embedding.pt"
        )
        download_url(
            url=url,
            folder=model_dir,
            filename=PREGENERATED_COLUMN_EMBEDDINGS_FILENAME,
        )
        print(f"Downloaded successfully from {url}")

    # Load the model from checkpoint
    loaded_model, config, criterion = load_model(
        path=Path(model_path),
        model_type=model_type,
    )
    loaded_model.cache_trainset_representation = False
    return loaded_model, config, criterion


def _remap_checkpoint_keys(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Map legacy flat checkpoint keys onto the new model/backbone split."""

    remapped: dict[str, torch.Tensor] = {}

    for key, value in state_dict.items():
        new_key = key
        if key.startswith("blocks."):
            new_key = f"transformer_encoder.{key}"
        elif key.startswith("feature_group_embedder."):
            new_key = f"x_pre_encoder.{key}"
        elif key.startswith("target_embedder."):
            new_key = f"y_pre_encoder.{key}"
        elif key.startswith("feature_positional_embedding_embeddings."):
            new_key = f"x_pre_encoder.{key}"
        elif key.startswith("pre_generated_column_embeddings"):
            new_key = f"x_pre_encoder.{key}"
        elif key == "add_thinking_rows.row_token_values_TE":
            new_key = "add_thinking_rows.row_token_values"
        remapped[new_key] = value

    return remapped
