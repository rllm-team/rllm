from __future__ import annotations
import math
import typing
import os
from pathlib import Path
from functools import partial
from collections.abc import Callable, Sequence
import numpy as np

import torch
import torch.nn as nn

from rllm.utils import download

#  Copyright (c) Prior Labs GmbH 2025.

# TODO: Seems like there's a lot in this file that is over-parametrized for regular
# usage. Could likely just remove it.

import functools
from functools import partial
from typing import Any, ClassVar, Optional
from .base import PerFeatureTransformer, ModelConfig
from .base import get_architecture, parse_config, create_inference_engine

from .encoders import (
    SequentialEncoder,
    SeqEncStep,
    RemoveEmptyFeaturesEncoderStep,
    RemoveDuplicateFeaturesEncoderStep,
    NanHandlingEncoderStep,
    InputNormalizationEncoderStep,
    VariableNumFeaturesEncoderStep,
    LinearInputEncoderStep,
    MulticlassClassificationTargetEncoder,
)
from .preprocessing import (
    EnsembleConfig,
    PreprocessorConfig,
    default_classifier_preprocessor_configs,
)
from .config import ModelInterfaceConfig

import torch
from torch import nn
from torch.nn.modules.transformer import Module, Tensor

HIDDEN_SIZE_LIMIT = 512
MLP_SAVE_PEAK_MEM_FACTOR = 32
MAXINT_RANDOM_SEED = int(np.iinfo(np.int32).max)


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
    inputs_to_merge = {"main": {"dim": num_features}}

    encoder_steps: list[SeqEncStep] = []
    if remove_empty_features:
        encoder_steps += [RemoveEmptyFeaturesEncoderStep()]

    if remove_duplicate_features:
        encoder_steps += [RemoveDuplicateFeaturesEncoderStep()]

    encoder_steps += [NanHandlingEncoderStep(keep_nans=nan_handling_enabled)]

    if nan_handling_enabled:
        inputs_to_merge["nan_indicators"] = {"dim": num_features}

        encoder_steps += [
            VariableNumFeaturesEncoderStep(
                num_features=num_features,
                normalize_by_used_features=False,
                in_keys=["nan_indicators"],
                out_keys=["nan_indicators"],
            ),
        ]

    encoder_steps += [
        InputNormalizationEncoderStep(
            normalize_on_train_only=normalize_on_train_only,
            normalize_to_ranking=normalize_to_ranking,
            normalize_x=normalize_x,
            remove_outliers=remove_outliers,
        ),
    ]

    encoder_steps += [
        VariableNumFeaturesEncoderStep(
            num_features=num_features,
            normalize_by_used_features=normalize_by_used_features,
        ),
    ]

    encoder_steps += [
        LinearInputEncoderStep(
            num_features=sum([i["dim"] for i in inputs_to_merge.values()]),
            emsize=embedding_size,
            bias=encoder_use_bias,
            in_keys=tuple(inputs_to_merge),
            out_keys=("output",),
        ),
    ]

    return SequentialEncoder(*encoder_steps, output_key="output")


def get_y_encoder(
    *,
    num_inputs: int,
    embedding_size: int,
    nan_handling_y_encoder: bool,
    max_num_classes: int,
) -> nn.Module:
    steps: list[SeqEncStep] = []
    inputs_to_merge = [{"name": "main", "dim": num_inputs}]
    if nan_handling_y_encoder:
        steps += [NanHandlingEncoderStep()]
        inputs_to_merge += [{"name": "nan_indicators", "dim": num_inputs}]

    if max_num_classes >= 2:
        steps += [MulticlassClassificationTargetEncoder()]

    steps += [
        LinearInputEncoderStep(
            num_features=sum([i["dim"] for i in inputs_to_merge]),  # type: ignore
            emsize=embedding_size,
            in_keys=tuple(i["name"] for i in inputs_to_merge),  # type: ignore
            out_keys=("output",),
        ),
    ]
    return SequentialEncoder(*steps, output_key="output")


def load_model_only_inference(path, filename, device):
    """
    Loads a saved model from the specified position. This function only restores inference capabilities and
    cannot be used for further training.
    """

    models = torch.load(os.path.join(path, filename), map_location="cpu")
    model_state = models["state_dict"]
    config_sample = models["config"]

    assert config_sample["max_num_classes"] > 2
    loss = torch.nn.CrossEntropyLoss(
        reduction="none", weight=torch.ones(int(config_sample["max_num_classes"]))
    )

    config = parse_config(config_sample)[0]

    if config.max_num_classes == 2:
        n_out = 1
    if config.max_num_classes > 2 and isinstance(loss, nn.CrossEntropyLoss):
        n_out = config.max_num_classes
    # if config["max_num_classes"] == 0 and isinstance(loss, BarDistribution):
    #     return loss.num_bars

    model = PerFeatureTransformer(
        config=config,
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
        cache_trainset_representation=True,
        use_encoder_compression_layer=False,
        n_out=n_out,
        #
        # These are things that had default values from config.get() but were not
        # present in any config.
        layer_norm_with_elementwise_affine=False,
    )

    # print(f"Using a Transformer with {sum(p.numel() for p in model.parameters()) / 1000 / 1000:.{2}f} M parameters")

    model.criterion = loss
    module_prefix = "module."
    model_state = {k.replace(module_prefix, ""): v for k, v in model_state.items()}
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()

    return (float("inf"), float("inf"), model), config_sample  # no loss measured


def load_model_workflow(
    i, e, add_name, base_path, device="cpu", eval_addition="", only_inference=True
):
    """
    Workflow for loading a model and setting appropriate parameters for diffable hparam tuning.

    :param i:
    :param e:
    :param eval_positions_valid:
    :param add_name:
    :param base_path:
    :param device:
    :param eval_addition:
    :return:
    """

    filenames = [
        "tabpfn-v2-classifier.ckpt",
        "tabpfn-v2-classifier-gn2p4bpt.ckpt",
        "tabpfn-v2-classifier-llderlii.ckpt",
        "tabpfn-v2-classifier-od3j1g5m.ckpt",
        "tabpfn-v2-classifier-vutqq28w.ckpt",
        "tabpfn-v2-classifier-znskzxi4.ckpt",
        "tabpfn-v2-classifier-finetuned-zk73skhh.ckpt",
        "tabpfn-v2-classifier-finetuned-znskzxi4-tvvss6bp.ckpt",
        "tabpfn-v2-classifier-finetuned-vutqq28w-boexhu6h.ckpt",
        "tabpfn-v2-classifier-finetuned-od3j1g5m-4svepuy5.ckpt",
        "tabpfn-v2-classifier-finetuned-llderlii-oyd7ul21.ckpt",
        "tabpfn-v2-classifier-finetuned-gn2p4bpt-xp6f0iqb.ckpt",
    ]

    def get_file(e):
        """
        Returns the different paths of model_file, model_path and results_file
        """
        model_file = (
            f"models_diff/prior_diff_real_checkpoint{add_name}_n_{i}_epoch_{e}.cpkt"
        )
        model_path = os.path.join(base_path, model_file)
        # print('Evaluate ', model_path)
        results_file = os.path.join(
            base_path,
            f"models_diff/prior_diff_real_results{add_name}_n_{i}_epoch_{e}_{eval_addition}.pkl",
        )
        return model_file, model_path, results_file

    def check_file(e):
        model_file, model_path, results_file = get_file(e)
        if not Path(model_path).is_file():  # or Path(results_file).is_file():
            print(
                "We have to download the TabPFN, as there is no checkpoint at ",
                model_path,
            )
            print("It has about 100MB, so this might take a moment.")
            import requests

            url = "https://github.com/PriorLabs/TabPFN/raw/refs/tags/v1.0.0/tabpfn/models_diff/prior_diff_real_checkpoint_n_0_epoch_42.cpkt"
            print("hhh")
            r = requests.get(url, allow_redirects=True)
            print("hhh")
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            open(model_path, "wb").write(r.content)
        return model_file, model_path, results_file

    model_file = None
    if e == -1:
        for e_ in range(100, -1, -1):
            model_file_, model_path_, results_file_ = check_file(e_)
            if model_file_ is not None:
                e = e_
                model_file, model_path, results_file = (
                    model_file_,
                    model_path_,
                    results_file_,
                )
                break
    else:
        model_file, model_path, results_file = check_file(e)

    if model_file is None:
        model_file, model_path, results_file = get_file(e)
        raise Exception("No checkpoint found at " + str(model_path))

    # print(f'Loading {model_file}')
    if only_inference:
        # print('Loading model that can be used for inference only')
        model, c = load_model_only_inference(base_path, model_file, device)
    """
    else:
        #until now also only capable of inference
        model, c = load_model(base_path, model_file, device, eval_positions=[], verbose=False)
    """
    # model, c = load_model(base_path, model_file, device, eval_positions=[], verbose=False)

    return model, c, results_file


def download_model(model_type, model_name, download_path):
    """
    Downloads a model file from two possible URLs, trying the second if the first fails.

    Args:
        model_type (str): The type of the model (e.g., "TabPFN-v2-clf").
        model_name (str): The specific model file name (e.g., "tabpfn-v2-classifier.ckpt").
        download_path (str): The folder where the model should be saved.

    Returns:
        bool: True if the download is successful, False otherwise.
    """
    repo_id = f"Prior-Labs/TabPFN-v2-{model_type}"
    urls = [
        f"https://huggingface.co/{repo_id}/resolve/main/{model_name}?download=true",
        f"https://hf-mirror.com/{repo_id}/resolve/main/{model_name}?download=true",
    ]
    print("111", repo_id)
    # Ensure the download path exists
    os.makedirs(download_path, exist_ok=True)
    for url in urls:
        try:
            # Attempt to download the file
            download.download_url(url=url, folder=download_path, filename=model_name)
            print(f"Downloaded successfully from {url}")
            return True
        except Exception as e:
            print(f"Failed to download from {url}: {e}")

    print("Download failed from both URLs.")
    return False


def get_filename_from_model_name(
    model_type: str,
    model_name: Optional[str] = None,
) -> str:
    classifier_filenames = [
        "tabpfn-v2-classifier.ckpt",
        "tabpfn-v2-classifier-gn2p4bpt.ckpt",
        "tabpfn-v2-classifier-llderlii.ckpt",
        "tabpfn-v2-classifier-od3j1g5m.ckpt",
        "tabpfn-v2-classifier-vutqq28w.ckpt",
        "tabpfn-v2-classifier-znskzxi4.ckpt",
        "tabpfn-v2-classifier-finetuned-zk73skhh.ckpt",
        "tabpfn-v2-classifier-finetuned-znskzxi4-tvvss6bp.ckpt",
        "tabpfn-v2-classifier-finetuned-vutqq28w-boexhu6h.ckpt",
        "tabpfn-v2-classifier-finetuned-od3j1g5m-4svepuy5.ckpt",
        "tabpfn-v2-classifier-finetuned-llderlii-oyd7ul21.ckpt",
        "tabpfn-v2-classifier-finetuned-gn2p4bpt-xp6f0iqb.ckpt",
    ]
    regressor_filenames = [
        "tabpfn-v2-regressor.ckpt",
        "tabpfn-v2-regressor-09gpqh39.ckpt",
        "tabpfn-v2-regressor-2noar4o2.ckpt",
        "tabpfn-v2-regressor-wyl4o83o.ckpt",
    ]
    if model_type == "clf":
        if model_name is None or model_name not in classifier_filenames:
            return classifier_filenames[0]
    elif model_type == "reg":
        if model_name is None or model_name not in regressor_filenames:
            return regressor_filenames[0]
    return model_name


def load_model_state(
    model_path,
    model_type,
    model_name: Optional[str] = None,
):

    file_name = get_filename_from_model_name(model_type, model_name)
    model_file_path = os.path.join(model_path, file_name)

    # Check if the model file exists
    if os.path.exists(model_file_path):
        print(f"Model file found locally: {model_file_path}")
    else:
        print(f"Model file not found. Downloading to: {model_file_path}")
        success = download_model(model_type, file_name, model_path)
        if not success:
            raise FileNotFoundError(f"Failed to download the model: {file_name}")

    # Load the model using the provided load function
    print(f"Loading model from: {model_file_path}")
    model_state = torch.load(model_file_path)
    return model_state


class TabPFN(torch.nn.Module):
    def __init__(
        self,
        model_path: str,
        model_type: str = "clf",
        random_state: int | np.random.RandomState | np.random.Generator | None = None,
    ):
        """
        model_type: clf or reg means "classifier" or "regressor"
        """
        super().__init__()
        model_config = load_model_state(
            model_path="models/tabpfnv2", model_type=model_type
        )
        static_seed, rng = infer_random_state(random_state)
        self.rng = rng
        self.static_seed = static_seed

        model_state = model_config["state_dict"]
        config_sample = model_config["config"]

        config = parse_config(config_sample)[0]
        device = "cuda" if torch.cuda.is_available() else "cpu:0"

        assert config_sample["max_num_classes"] > 2
        loss = torch.nn.CrossEntropyLoss(
            reduction="none", weight=torch.ones(int(config_sample["max_num_classes"]))
        )

        n_out = 2
        if config.max_num_classes == 2:
            n_out = 1
        if config.max_num_classes > 2 and isinstance(loss, nn.CrossEntropyLoss):
            n_out = config.max_num_classes
        # if config.max_num_classes == 0 and isinstance(loss, BarDistribution):
        #     n_out = loss.num_bars
        model = get_architecture(
            config=config,
            n_out=n_out,
            cache_trainset_representation=False,
        )

        # model.criterion = loss
        module_prefix = "module."
        model_state = {k.replace(module_prefix, ""): v for k, v in model_state.items()}

        model.load_state_dict(model_state)
        model.to(device)
        model.eval()

        self.n_estimators = 8
        self.interface_config_ = ModelInterfaceConfig.from_user_input(
            inference_config=None,
        )
        self.differentiable_input = False
        self.n_classes_ = 2
        self.byte_size = 4
        self.n_jobs = 1
        self.memory_saving_mode = "auto"
        self.forced_inference_dtype_ = None
        self.use_autocast_ = False

        self.model = model

    def fit(self, x: torch.Tensor, y: torch.Tensor):
        self.ensemble_configs = self._init_dataset_preprocessing(x, y)
        self.executor_ = create_inference_engine(
            X_train=x,
            y_train=y,
            model=self.model,
            ensemble_configs=self.ensemble_configs,
            cat_ix=self.inferred_categorical_indices_,
            devices_="cpu",
            rng=self.rng,
            n_jobs=self.n_jobs,
            byte_size=self.byte_size,
            forced_inference_dtype_=self.forced_inference_dtype_,
            memory_saving_mode=self.memory_saving_mode,
            use_autocast_=self.use_autocast_,
            inference_mode=not self.differentiable_input,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        print(self.ensemble_configs)
        return self.model(x, categorical_inds=[[6, 8]])

    def _init_dataset_preprocessing(self, X, y):
        self.n_features_in_ = 7
        self.inferred_categorical_indices_ = [0, 1, 6]
        preprocess_transforms = self.interface_config_.PREPROCESS_TRANSFORMS
        ensemble_configs = EnsembleConfig.generate_for_classification(
            num_estimators=self.n_estimators,
            subsample_size=self.interface_config_.SUBSAMPLE_SAMPLES,
            add_fingerprint_feature=self.interface_config_.FINGERPRINT_FEATURE,
            feature_shift_decoder=self.interface_config_.FEATURE_SHIFT_METHOD,
            polynomial_features=self.interface_config_.POLYNOMIAL_FEATURES,
            max_index=len(X),
            preprocessor_configs=typing.cast(
                "Sequence[PreprocessorConfig]",
                (
                    preprocess_transforms
                    if preprocess_transforms is not None
                    else default_classifier_preprocessor_configs()
                ),
            ),
            class_shift_method=(
                self.interface_config_.CLASS_SHIFT_METHOD
                if not self.differentiable_input
                else None
            ),
            n_classes=self.n_classes_,
            random_state=self.rng,
            num_models=1,
        )
        return ensemble_configs


def infer_random_state(
    random_state: int | np.random.RandomState | np.random.Generator | None,
) -> tuple[int, np.random.Generator]:
    """Infer the random state from the given input.

    Args:
        random_state: The random state to infer.

    Returns:
        A static integer seed and a random number generator.
    """
    if isinstance(random_state, (int, np.integer)):
        np_rng = np.random.default_rng(random_state)
        static_seed = int(random_state)
    elif isinstance(random_state, np.random.RandomState):
        static_seed = int(random_state.randint(0, MAXINT_RANDOM_SEED))
        np_rng = np.random.default_rng(static_seed)
    elif isinstance(random_state, np.random.Generator):
        np_rng = random_state
        static_seed = int(np_rng.integers(0, MAXINT_RANDOM_SEED))
    elif random_state is None:
        np_rng = np.random.default_rng()
        static_seed = int(np_rng.integers(0, MAXINT_RANDOM_SEED))
    else:
        raise ValueError(f"Invalid random_state {random_state}")

    return static_seed, np_rng
