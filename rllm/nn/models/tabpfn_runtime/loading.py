#  Copyright (c) Prior Labs GmbH 2025.

from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn

from rllm.nn.loss import FullSupportBarDistribution
from rllm.utils import download_model_from_huggingface, download_url

from .backbone import TabPFNModel
from .utils import PREGENERATED_COLUMN_EMBEDDINGS_FILENAME

logger = logging.getLogger(__name__)
ModelVersion = Literal["v2_6"]

V2_6_NLAYERS = 24
V2_6_FEATURE_GROUP_SIZE = 3
V2_6_NUM_THINKING_ROWS = 64
V2_6_REGRESSION_MLP_HIDDEN_DIM = 1024


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


def get_filename_from_model_name(
    model_type: str,
    model_id: int | None = None,
    *,
    version: ModelVersion = "v2_6",
) -> str:
    if version != "v2_6":
        raise ValueError(
            f"Unsupported TabPFN version: {version}. Only 'v2_6' is available."
        )
    model_filenames = {
        "clf": [
            "tabpfn-v2.6-classifier-v2.6_default.ckpt",
        ],
        "reg": [
            "tabpfn-v2.6-regressor-v2.6_default.ckpt",
        ],
    }

    filenames = model_filenames.get(model_type)
    if filenames is None:
        raise ValueError(f"Invalid model_type: {model_type}")
    if model_id is None or model_id not in range(len(filenames)):
        return filenames[0]
    return filenames[model_id]


def get_hf_repo_from_model_name(
    *,
    version: ModelVersion = "v2_6",
) -> str:
    if version != "v2_6":
        raise ValueError(
            f"Unsupported TabPFN version: {version}. Only 'v2_6' is available."
        )
    return "Prior-Labs/tabpfn_2_6"


def load_model_criterion_config(
    model_dir: str,
    model_type: str,
    model_id: int | None = None,
    *,
    cache_trainset_representation: bool,
    version: ModelVersion = "v2_6",
    strict_version_match: bool = True,
) -> tuple[torch.nn.Module, dict[str, Any], Any]:
    """Load the model, criterion, and config, downloading the checkpoint if needed."""

    model_name = get_filename_from_model_name(
        model_type,
        model_id,
        version=version,
    )
    model_path = os.path.join(model_dir, model_name)

    print(f"Loading model from {model_dir}...")
    if not os.path.exists(model_path):
        download_model_from_huggingface(
            repo=get_hf_repo_from_model_name(version=version),
            model_name=model_name,
            download_path=model_dir,
        )

    column_embedding_path = os.path.join(
        model_dir,
        PREGENERATED_COLUMN_EMBEDDINGS_FILENAME,
    )
    if not os.path.exists(column_embedding_path):
        try:
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
        except Exception as e:  # noqa: BLE001
            print(
                "Optional companion file was not downloaded: "
                f"{PREGENERATED_COLUMN_EMBEDDINGS_FILENAME}. "
                "The checkpoint download itself succeeded. "
                f"Error: {e}"
            )

    loaded_model, config, criterion = load_model(
        path=Path(model_path),
        model_type=model_type,
        version=version,
        strict_version_match=strict_version_match,
    )
    loaded_model.cache_trainset_representation = cache_trainset_representation
    return loaded_model, config, criterion


def initialize_tabpfn_model(
    model_dir: str,
    model_type: str,
    model_id: int,
    version: ModelVersion = "v2_6",
    strict_version_match: bool = True,
) -> tuple[torch.nn.Module, dict[str, Any], object]:
    """Load the TabPFN model, config, and regression criterion when applicable."""
    if model_type == "clf":
        return load_model_criterion_config(
            model_dir=model_dir,
            model_type=model_type,
            model_id=model_id,
            cache_trainset_representation=False,
            version=version,
            strict_version_match=strict_version_match,
        )

    return load_model_criterion_config(
        model_dir=model_dir,
        model_type=model_type,
        model_id=model_id,
        cache_trainset_representation=False,
        version=version,
        strict_version_match=strict_version_match,
    )


def load_model(
    *,
    path: Path,
    model_type: Literal["clf", "reg"],
    version: ModelVersion = "v2_6",
    strict_version_match: bool = True,
) -> tuple[
    nn.Module,
    dict[str, Any],
    object,
]:
    """Load a retained TabPFN v2.6 checkpoint into the local runtime."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        checkpoint = torch.load(path, map_location="cpu", weights_only=None)
    state_dict = checkpoint["state_dict"]
    config = checkpoint.get("config")

    inferred_model_version = _infer_model_version_from_path(path)
    if version != inferred_model_version:
        message = (
            "Checkpoint/runtime version mismatch: "
            f"requested={version}, "
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
        version = inferred_model_version

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
        nlayers=V2_6_NLAYERS,
        nhead=int(config["nhead"]),
        features_per_group=V2_6_FEATURE_GROUP_SIZE,
        num_thinking_rows=V2_6_NUM_THINKING_ROWS,
        encoder_type="mlp" if task_type == "regression" else "linear",
        encoder_mlp_hidden_dim=V2_6_REGRESSION_MLP_HIDDEN_DIM,
    )
    pregenerated_path = path.parent / PREGENERATED_COLUMN_EMBEDDINGS_FILENAME
    if not pregenerated_path.exists():
        raise FileNotFoundError(
            f"Required companion file '{pregenerated_path.name}' was not found "
            f"next to checkpoint '{path.name}'."
        )

    model = TabPFNModel(
        **model_config_kwargs,
        n_out=n_out,
        task_type=task_type,
        column_embeddings_path=pregenerated_path,
    )

    load_checkpoint_compatible(
        model=model,
        state_dict=state_dict,
        source_checkpoint_version=inferred_model_version,
        source_model_version=inferred_model_version,
        target_model_version=version,
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
