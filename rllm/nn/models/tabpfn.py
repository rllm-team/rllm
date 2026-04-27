from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
from typing import Optional, List, Literal, Any
import numpy as np
import os
from pathlib import Path
import warnings

import torch

from rllm.data_augment import TabPFNEnsembleAugmentor
from .tabpfn_internal.config import (
    ExportMode,
    ModelVersion,
    TabPFNVersionConfig,
    build_version_config,
)
from .tabpfn_internal.export import (
    ExportArtifacts,
    export_as_mlp_stub,
    export_as_tree_ensemble_stub,
)
from .tabpfn_internal.input_cleaning import (
    clean_data as clean_tabpfn_data,
    fix_dtypes as fix_tabpfn_dtypes,
    process_text_na_dataframe,
)
from .tabpfn_internal.loading import load_checkpoint_compatible, load_model
from .tabpfn_internal.postprocess import ClassifierPostProcessor
from rllm.types import ColType
from rllm.utils import download_model_from_huggingface


def _map_to_bucket_ix(y: torch.Tensor, borders: torch.Tensor) -> torch.Tensor:
    ix = torch.searchsorted(sorted_sequence=borders, input=y) - 1
    ix[y == borders[0]] = 0
    ix[y == borders[-1]] = len(borders) - 2
    return ix


def _cdf(logits: torch.Tensor, borders: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
    ys = ys.repeat((*logits.shape[:-1], 1))
    n_bars = len(borders) - 1
    y_buckets = _map_to_bucket_ix(ys, borders).clamp(0, n_bars - 1).to(logits.device)

    probs = torch.softmax(logits, dim=-1)
    prob_so_far = torch.cumsum(probs, dim=-1) - probs
    prob_left_of_bucket = prob_so_far.gather(index=y_buckets, dim=-1)

    bucket_widths = borders[1:] - borders[:-1]
    share_of_bucket_left = (ys - borders[y_buckets]) / bucket_widths[y_buckets]
    share_of_bucket_left = share_of_bucket_left.clamp(0.0, 1.0)

    prob_in_bucket = probs.gather(index=y_buckets, dim=-1) * share_of_bucket_left
    prob_left_of_ys = prob_left_of_bucket + prob_in_bucket

    prob_left_of_ys[ys <= borders[0]] = 0.0
    prob_left_of_ys[ys >= borders[-1]] = 1.0
    return prob_left_of_ys.clip(0.0, 1.0)


def _translate_probs_across_borders(
    logits: torch.Tensor,
    *,
    frm: torch.Tensor,
    to: torch.Tensor,
) -> torch.Tensor:
    prob_left = _cdf(logits, borders=frm, ys=to)
    prob_left[..., 0] = 0.0
    prob_left[..., -1] = 1.0
    return (prob_left[..., 1:] - prob_left[..., :-1]).clamp_min(0.0)


def get_filename_from_model_name(
    model_type: str,
    model_id: Optional[int] = None,
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
    model_type: str,
    *,
    version: ModelVersion = "v2_6",
) -> str:
    del model_type
    if version != "v2_6":
        raise ValueError(
            f"Unsupported TabPFN version: {version}. Only 'v2_6' is available."
        )
    return "Prior-Labs/tabpfn_2_6"


def load_model_criterion_config(
    model_dir: str,
    model_type: str,
    model_id: Optional[int] = None,
    *,
    check_bar_distribution_criterion: bool,
    cache_trainset_representation: bool,
    model_seed: int,
    metadata: Optional[dict[ColType, list[dict[Any, Any]]]] = None,
    version_config: TabPFNVersionConfig | None = None,
    strict_version_match: bool = True,
) -> tuple[torch.nn.Module, dict[str, Any], Any]:
    """Load the model, criterion, and config from the given path.

    Args:
        model_path: The path to the model.
        check_bar_distribution_criterion:
            Whether to check if the criterion
            is a FullSupportBarDistribution, which is the expected criterion
            for models trained for regression.
        cache_trainset_representation:
            Whether the model should know to cache the trainset representation.
        which: Whether the model is a regressor or classifier.
        version: The version of the model.
        download: Whether to download the model if it doesn't exist.
        model_seed: The seed of the model.
        metadata: Optional column metadata passed through ``load_model``.

    Returns:
        The model, criterion, and config.
    """

    model_version = (
        version_config.model_version if version_config is not None else "v2_6"
    )
    model_name = get_filename_from_model_name(
        model_type,
        model_id,
        version=model_version,
    )
    model_path = os.path.join(model_dir, model_name)

    print(f"Loading model from {model_dir}...")
    if not os.path.exists(model_path):
        download_model_from_huggingface(
            repo=get_hf_repo_from_model_name(model_type, version=model_version),
            model_name=model_name,
            download_path=model_dir,
        )
    loaded_model, config, criterion = load_model(
        path=Path(model_path),
        model_seed=model_seed,
        model_type=model_type,
        metadata=metadata,
        version_config=version_config,
        strict_version_match=strict_version_match,
    )
    loaded_model.cache_trainset_representation = cache_trainset_representation
    return loaded_model, config, criterion


def initialize_tabpfn_model(
    model_dir: str,
    model_type: str,
    model_id: int,
    static_seed: int,
    metadata: Optional[dict[ColType, list[dict[Any, Any]]]] = None,
    version_config: TabPFNVersionConfig | None = None,
    strict_version_match: bool = True,
) -> tuple[torch.nn.Module, dict[str, Any], object]:
    """Common logic to load the TabPFN model, set up the random state,
    and optionally download the model.

    Args:
        model_path: Path or directive ("auto") to load the pre-trained model from.
        which: Which TabPFN model to load.
        fit_mode: Determines caching behavior.
        static_seed: Random seed for reproducibility logic.
        metadata: Optional column metadata for the pre-encoder (see ``load_model``).

    Returns:
        model: The loaded TabPFN model.
        config: The configuration object associated with the loaded model.
        criterion: FullSupportBarDistribution for regression, None for classification.
    """

    # Load model with potential caching
    if model_type == "clf":
        # The classifier's bar distribution is not used;
        # pass check_bar_distribution_criterion=False
        model, config_, criterion = load_model_criterion_config(
            model_dir=model_dir,
            model_type=model_type,
            model_id=model_id,
            check_bar_distribution_criterion=False,
            cache_trainset_representation=False,
            model_seed=static_seed,
            metadata=metadata,
            version_config=version_config,
            strict_version_match=strict_version_match,
        )
    else:
        # The regressor's bar distribution is required
        model, config_, criterion = load_model_criterion_config(
            model_dir=model_dir,
            model_type=model_type,
            model_id=model_id,
            check_bar_distribution_criterion=True,
            cache_trainset_representation=False,
            model_seed=static_seed,
            metadata=metadata,
            version_config=version_config,
            strict_version_match=strict_version_match,
        )

    return model, config_, criterion


class TabPFN(torch.nn.Module):
    """
    TabPFN model wrapper for classification and regression tasks.

    This class wraps the TabPFN model and provides a PyTorch nn.Module interface
    for easy integration with PyTorch workflows.

    Args:
        model_dir (str): Directory containing the TabPFN model checkpoint.
        model_type (str): Type of model, either 'clf' for classification or 'reg' for regression.
            Default: 'clf'.
        model_id (int): Model ID to load. Default: 0.
        static_seed (int): Random seed for reproducibility. Default: 0.
        n_estimators (int): Number of ensemble estimators. Default: 4.
        subsample_size (int | None): Optional row subsample size for each
            estimator. If ``None``, no row subsampling is applied, matching the
            retained official inference pipeline.
        add_fingerprint_feature (bool): Whether to add fingerprint features. Default: True.
        feature_shift_decoder (str): Feature shift decoder strategy. Default: 'shuffle'.
        polynomial_features (str | int): Polynomial features strategy.
            The retained regression preset defaults to ``10``.
        class_shift_method (str): Class shift method. Default: 'shuffle'.
        softmax_temperature (float): Temperature for softmax scaling. Default: 0.9.
        average_before_softmax (bool): Whether to average logits before softmax. Default: False.
        n_workers (int): Number of parallel workers for preprocessing. Default: 1.
        parallel_mode (str): Parallel processing mode. Default: 'block'.
        metadata: Optional column metadata carried through model loading.
    """

    def __init__(
        self,
        model_dir: str,
        model_type: Literal["clf", "reg"] = "clf",
        model_id: int = 0,
        static_seed: int = 0,
        n_estimators: int = 4,
        subsample_size: int | None = None,
        add_fingerprint_feature: bool = True,
        feature_shift_decoder: str = "shuffle",
        polynomial_features: Literal["no", "all"] | int = "no",
        class_shift_method: str = "shuffle",
        softmax_temperature: float = 0.9,
        average_before_softmax: bool = False,
        n_workers: int = 1,
        parallel_mode: str = "block",
        metadata: Optional[dict[ColType, list[dict[Any, Any]]]] = None,
        version: ModelVersion = "v2_6",
        n_layers_cls: int | None = None,
        n_layers_reg: int | None = None,
        feature_group_size: int | None = None,
        use_regression_mlp_encoder: bool | None = None,
        regression_mlp_hidden_dim: int | None = None,
        use_thinking_rows: bool | None = None,
        n_thinking_rows: int | None = None,
        preprocessing_recipe: str | None = None,
        use_robust_scaling: bool | None = None,
        use_soft_clipping: bool | None = None,
        use_quantile_transform: bool | None = None,
        use_standard_scaling: bool | None = None,
        enable_temperature_scaling: bool | None = None,
        enable_threshold_tuning: bool | None = None,
        export_mode: ExportMode = "icl",
        enable_flash_attention: bool = False,
        inference_batch_size: int = 4096,
        strict_version_match: bool = True,
    ):
        super().__init__()

        # Store hyperparameters
        self.model_dir = model_dir
        self.model_type = model_type
        self.model_id = model_id
        self.static_seed = static_seed
        self.n_estimators = n_estimators
        self.add_fingerprint_feature = add_fingerprint_feature
        self.feature_shift_decoder = feature_shift_decoder
        self.polynomial_features = polynomial_features
        self.class_shift_method = class_shift_method
        self.softmax_temperature = softmax_temperature
        self.average_before_softmax = average_before_softmax
        self.n_workers = n_workers
        self.parallel_mode = parallel_mode
        self.inference_batch_size = int(inference_batch_size)
        self.strict_version_match = bool(strict_version_match)
        preset = TabPFNVersionConfig.from_version(version)
        if n_layers_cls is None:
            n_layers_cls = preset.n_layers_cls
        if n_layers_reg is None:
            n_layers_reg = preset.n_layers_reg
        if feature_group_size is None:
            feature_group_size = preset.feature_group_size
        if use_regression_mlp_encoder is None:
            use_regression_mlp_encoder = preset.use_regression_mlp_encoder
        if regression_mlp_hidden_dim is None:
            regression_mlp_hidden_dim = preset.regression_mlp_hidden_dim
        if use_thinking_rows is None:
            use_thinking_rows = preset.use_thinking_rows
        if n_thinking_rows is None:
            n_thinking_rows = preset.n_thinking_rows
        if preprocessing_recipe is None:
            preprocessing_recipe = preset.preprocessing_recipe
        if use_robust_scaling is None:
            use_robust_scaling = preset.use_robust_scaling
        if use_soft_clipping is None:
            use_soft_clipping = preset.use_soft_clipping
        if use_quantile_transform is None:
            use_quantile_transform = preset.use_quantile_transform
        if use_standard_scaling is None:
            use_standard_scaling = preset.use_standard_scaling
        if enable_temperature_scaling is None:
            enable_temperature_scaling = preset.enable_temperature_scaling
        if enable_threshold_tuning is None:
            enable_threshold_tuning = preset.enable_threshold_tuning
        if export_mode == "icl":
            export_mode = preset.export_mode
        if polynomial_features == "no" and model_type == "reg" and version == "v2_6":
            polynomial_features = 10
        self.polynomial_features = polynomial_features

        self.version_config = build_version_config(
            model_type,
            model_version=version,
            n_layers_cls=n_layers_cls,
            n_layers_reg=n_layers_reg,
            feature_group_size=feature_group_size,
            use_regression_mlp_encoder=use_regression_mlp_encoder,
            regression_mlp_hidden_dim=regression_mlp_hidden_dim,
            use_thinking_rows=use_thinking_rows,
            n_thinking_rows=n_thinking_rows,
            preprocessing_recipe=preprocessing_recipe,
            use_robust_scaling=use_robust_scaling,
            use_soft_clipping=use_soft_clipping,
            use_quantile_transform=use_quantile_transform,
            use_standard_scaling=use_standard_scaling,
            enable_temperature_scaling=enable_temperature_scaling,
            enable_threshold_tuning=enable_threshold_tuning,
            export_mode=export_mode,
            enable_flash_attention=enable_flash_attention,
        )
        self.subsample_size = self._resolve_default_subsample_size(subsample_size)
        self.postprocessor = ClassifierPostProcessor()
        self._fit_cat_ix: list[int] = []
        self._ordinal_encoder = None
        self._X_train_fit: np.ndarray | None = None
        self._y_train_fit: np.ndarray | None = None
        self._member_models: list[torch.nn.Module] = []
        self.y_train_mean_ = 0.0
        self.y_train_std_ = 1.0
        self.znorm_space_bardist_ = None
        self.raw_space_bardist_ = None

        # Initialize model
        self.model, self.config, self.bardist = initialize_tabpfn_model(
            model_dir=model_dir,
            model_type=model_type,
            model_id=model_id,
            static_seed=static_seed,
            metadata=metadata,
            version_config=self.version_config,
            strict_version_match=self.strict_version_match,
        )
        self.model.cache_trainset_representation = False
        if self.bardist is not None:
            self.znorm_space_bardist_ = deepcopy(self.bardist).float()
            self.raw_space_bardist_ = deepcopy(self.bardist).float()

        self.ensemble_augmentor: TabPFNEnsembleAugmentor | None = None
        # Fitted ensemble results produced by TabPFNEnsembleAugmentor.fit().
        self.ensemble_results = None
        self.n_classes = None

    def _resolve_default_subsample_size(self, subsample_size: int | None) -> int | None:
        """Resolve the default row subsampling used by the retained path."""
        if subsample_size is not None:
            return int(subsample_size)
        return None

    def _resolve_fit_subsample_size(self, n_samples: int) -> int | None:
        """Return the effective per-estimator row subsample size for fit()."""
        if self.subsample_size is None:
            return None
        return min(int(self.subsample_size), int(n_samples))

    def _apply_classification_temperature(
        self,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        if self.version_config.enable_temperature_scaling:
            logits_np = self.postprocessor.apply_temperature(
                logits.detach().float().cpu().numpy()
            )
            return torch.as_tensor(
                logits_np,
                dtype=torch.float32,
                device=logits.device,
            )
        if self.softmax_temperature != 1 and self.n_classes is not None:
            return logits / float(self.softmax_temperature)
        return logits

    def _classification_logits_to_probabilities(
        self,
        raw_logits: torch.Tensor,
    ) -> torch.Tensor:
        if raw_logits.ndim < 2:
            raise ValueError(
                f"Expected logits with 2 or more dims, got {raw_logits.ndim}"
            )

        logits = raw_logits.float()
        if self.version_config.enable_temperature_scaling:
            if logits.ndim >= 3:
                logits = torch.stack(
                    [
                        self._apply_classification_temperature(member_logits)
                        for member_logits in logits
                    ],
                    dim=0,
                )
            else:
                logits = self._apply_classification_temperature(logits)
        elif self.softmax_temperature != 1 and self.n_classes is not None:
            logits = logits / float(self.softmax_temperature)

        if logits.ndim >= 3:
            if (
                self.average_before_softmax
                or self.version_config.enable_temperature_scaling
            ):
                probs = torch.nn.functional.softmax(logits.mean(dim=0), dim=-1)
            else:
                probs = torch.nn.functional.softmax(logits, dim=-1).mean(dim=0)
        else:
            probs = torch.nn.functional.softmax(logits, dim=-1)

        return probs / probs.sum(dim=-1, keepdim=True)

    def fit(
        self,
        X_train: np.ndarray | torch.Tensor,
        y_train: np.ndarray | torch.Tensor,
        cat_ix: Optional[List[int]] = None,
        random_state: int = 0,
    ):
        """
        Prepare ensemble configurations for inference.

        Args:
            X_train: Training features of shape (n_samples, n_features).
            y_train: Training labels of shape (n_samples,).
            cat_ix: List of categorical feature indices. Default: None.
            random_state: Random state for reproducibility. Default: 0.

        Returns:
            self: The fitted model.
        """
        if cat_ix is None:
            cat_ix = []

        if isinstance(X_train, torch.Tensor):
            X_train = X_train.detach().cpu().numpy()
        if isinstance(y_train, torch.Tensor):
            y_train = y_train.detach().cpu().numpy()

        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train).reshape(-1)
        X_train_raw = X_train
        X_train, self._ordinal_encoder = clean_tabpfn_data(X_train, cat_ix)

        if self.model_type == "clf":
            y_train = y_train.astype(np.int64, copy=False)
            self.n_classes = len(np.unique(y_train))
            ensemble_augmentor = TabPFNEnsembleAugmentor(
                n_estimators=self.n_estimators,
                task="classification",
                subsample_size=self._resolve_fit_subsample_size(len(X_train)),
                add_fingerprint_feature=self.add_fingerprint_feature,
                feature_shift_decoder=self.feature_shift_decoder,
                polynomial_features=self.polynomial_features,
                class_shift_method=self.class_shift_method,
                random_state=random_state,
                max_index=len(X_train),
                n_classes=int(y_train.max()) + 1,
            )
            self.ensemble_results = list(
                ensemble_augmentor.fit(
                    X_train=X_train,
                    y_train=y_train,
                    cat_ix=cat_ix,
                )
            )
            self.ensemble_augmentor = ensemble_augmentor
            if self.version_config.enable_temperature_scaling:
                self._fit_temperature_scaling(X_train_raw, y_train, cat_ix)
            if self.version_config.enable_threshold_tuning:
                self._fit_threshold_tuning(X_train_raw, y_train, cat_ix)
        else:
            y_train = y_train.astype(np.float32, copy=False)
            self.y_train_mean_ = float(np.mean(y_train))
            self.y_train_std_ = float(np.std(y_train)) + 1e-20
            y_train = (y_train - self.y_train_mean_) / self.y_train_std_
            self.znorm_space_bardist_ = deepcopy(self.bardist).float()
            scaled_borders = (
                self.znorm_space_bardist_.borders * self.y_train_std_
                + self.y_train_mean_
            )
            self.raw_space_bardist_ = self.bardist.__class__(scaled_borders).float()
            ensemble_augmentor = TabPFNEnsembleAugmentor(
                n_estimators=self.n_estimators,
                task="regression",
                subsample_size=self._resolve_fit_subsample_size(len(X_train)),
                add_fingerprint_feature=self.add_fingerprint_feature,
                feature_shift_decoder=self.feature_shift_decoder,
                polynomial_features=self.polynomial_features,
                random_state=random_state,
                max_index=len(X_train),
            )
            self.ensemble_results = list(
                ensemble_augmentor.fit(
                    X_train=X_train,
                    y_train=y_train,
                    cat_ix=cat_ix,
                )
            )
            self.ensemble_augmentor = ensemble_augmentor
        self._fit_cat_ix = list(cat_ix)
        self._X_train_fit = X_train
        self._y_train_fit = y_train
        self._member_models = self._build_member_models()
        return self

    def _clean_predict_inputs(self, X: np.ndarray | torch.Tensor) -> np.ndarray:
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        X = np.asarray(X)
        X_df = fix_tabpfn_dtypes(X, self._fit_cat_ix)
        return process_text_na_dataframe(
            X=X_df,
            ord_encoder=self._ordinal_encoder,
            fit_encoder=False,
        )

    def _sdp_context(self):
        if not self.version_config.enable_flash_attention:
            return nullcontext()
        if not torch.cuda.is_available():
            return nullcontext()
        try:
            return torch.backends.cuda.sdp_kernel(
                enable_flash=True,
                enable_math=True,
                enable_mem_efficient=True,
            )
        except Exception:
            warnings.warn(
                "Flash attention kernel selection failed; falling back to default SDPA.",
                stacklevel=2,
            )
            return nullcontext()

    def _build_member_models(self) -> list[torch.nn.Module]:
        if self.ensemble_results is None:
            return []

        member_models: list[torch.nn.Module] = []
        base_device = next(self.model.parameters()).device
        for X_train_proc, y_train_proc, cat_ix_proc in self.ensemble_results:
            member_model = deepcopy(self.model)
            member_model.cache_trainset_representation = (
                self.model.cache_trainset_representation
            )
            member_model = member_model.to(base_device)
            if not member_model.cache_trainset_representation:
                member_model = member_model.cpu()
                member_models.append(member_model)
                continue
            X_train_tensor = torch.as_tensor(
                X_train_proc, dtype=torch.float32, device=base_device
            ).unsqueeze(1)
            y_train_tensor = torch.as_tensor(
                y_train_proc, dtype=torch.float32, device=base_device
            )

            with torch.inference_mode(), self._sdp_context():
                member_model(
                    X_train_tensor,
                    y_train_tensor,
                    only_return_standard_out=True,
                    categorical_inds=cat_ix_proc,
                    single_eval_pos=len(y_train_proc),
                )
            member_model = member_model.cpu()
            member_models.append(member_model)

        return member_models

    def _get_augmentor_pipelines(self) -> list[Any]:
        if self.ensemble_augmentor is None:
            raise RuntimeError(
                "Model must be fitted before prediction. Call fit() first."
            )
        pipelines = self.ensemble_augmentor.augmentor_pipelines
        if self.ensemble_results is not None and len(pipelines) != len(
            self.ensemble_results
        ):
            raise RuntimeError(
                "Fitted augmentor pipelines are missing or stale. Call fit() again."
            )
        return pipelines

    def _get_ensemble_configs(self) -> list[Any]:
        if self.ensemble_augmentor is None:
            raise RuntimeError(
                "Model must be fitted before prediction. Call fit() first."
            )
        configs = self.ensemble_augmentor.configs
        if self.ensemble_results is not None and len(configs) != len(
            self.ensemble_results
        ):
            raise RuntimeError(
                "Fitted ensemble configs are missing or stale. Call fit() again."
            )
        return configs

    def _predict_single_member(
        self,
        cfg: Any,
        augmentor_pipeline: Any,
        member_model: torch.nn.Module,
        X_train_proc: np.ndarray,
        y_train_proc: np.ndarray,
        cat_ix_proc: list[int],
        X_test: np.ndarray,
        *,
        return_debug: bool = False,
    ) -> torch.Tensor:
        device = next(self.model.parameters()).device
        X_test_proc, _ = augmentor_pipeline.transform(X_test)
        X_test_tensor = torch.as_tensor(X_test_proc, dtype=torch.float32, device=device)
        member_model = member_model.to(device)
        only_return_standard_out = not return_debug

        if member_model.cache_trainset_representation:
            with torch.inference_mode(), self._sdp_context():
                output = member_model(
                    X_test_tensor.unsqueeze(1),
                    None,
                    only_return_standard_out=only_return_standard_out,
                    categorical_inds=cat_ix_proc,
                )
        else:
            X_train_tensor = torch.as_tensor(
                X_train_proc, dtype=torch.float32, device=device
            )
            y_train_tensor = torch.as_tensor(
                y_train_proc, dtype=torch.float32, device=device
            )
            X_full = torch.cat([X_train_tensor, X_test_tensor], dim=0).unsqueeze(1)
            y_train_tensor = y_train_tensor.unsqueeze(1)
            with torch.inference_mode(), self._sdp_context():
                output = member_model(
                    X_full,
                    y_train_tensor,
                    only_return_standard_out=only_return_standard_out,
                    categorical_inds=cat_ix_proc,
                    single_eval_pos=len(y_train_proc),
                )
        if return_debug:
            if isinstance(output, dict):
                standard = output["standard"].squeeze(1)
                output["standard"] = standard
                return output
            raise RuntimeError("Expected debug forward output to be a dictionary.")
        output = output.squeeze(1)
        if self.model_type == "reg":
            if self.softmax_temperature != 1:
                output = output / self.softmax_temperature
            if cfg.target_transform is None:
                probs = _translate_probs_across_borders(
                    output,
                    frm=self.znorm_space_bardist_.borders.to(output.device),
                    to=self.znorm_space_bardist_.borders.to(output.device),
                )
                return self.raw_space_bardist_.mean(probs.log()).float()
            mean_pred = self.znorm_space_bardist_.mean(output).float()
            if cfg.target_transform is not None:
                mean_np = mean_pred.cpu().numpy().reshape(-1, 1)
                mean_np = cfg.target_transform.inverse_transform(mean_np).ravel()
                mean_pred = torch.as_tensor(mean_np, device=device, dtype=torch.float32)
            return mean_pred * self.y_train_std_ + self.y_train_mean_

        if cfg.class_permutation is not None:
            output = output[..., cfg.class_permutation]
        return output

    def forward(
        self,
        X_test: np.ndarray | torch.Tensor,
        *,
        return_logits: bool = False,
    ) -> np.ndarray:
        """
        Forward pass through the TabPFN model with ensemble inference.

        For classification (model_type='clf'), returns predicted probabilities
        of shape (n_test, n_classes).

        For regression (model_type='reg'), returns predicted values
        of shape (n_test,).

        Args:
            X_test: Test features of shape (n_test, n_features).

        Returns:
            Predictions array.
        """
        if self.ensemble_results is None:
            raise RuntimeError(
                "Model must be fitted before prediction. Call fit() first."
            )
        if len(self._member_models) != len(self.ensemble_results):
            raise RuntimeError(
                "Cached ensemble member models are missing or stale. Call fit() again."
            )

        X_test = self._clean_predict_inputs(X_test)
        outputs: list[torch.Tensor] = []

        # Iterate through all preprocessing configurations
        for member_model, augmentor_pipeline, cfg, (
            X_train_proc,
            y_train_proc,
            cat_ix_proc,
        ) in zip(
            self._member_models,
            self._get_augmentor_pipelines(),
            self._get_ensemble_configs(),
            self.ensemble_results,
        ):
            member_batches = []
            for start in range(0, len(X_test), self.inference_batch_size):
                end = min(len(X_test), start + self.inference_batch_size)
                member_batches.append(
                    self._predict_single_member(
                        cfg,
                        augmentor_pipeline,
                        member_model,
                        X_train_proc,
                        y_train_proc,
                        cat_ix_proc,
                        X_test[start:end],
                    )
                )
            output = torch.cat(member_batches, dim=0)
            outputs.append(output)

        if self.model_type == "reg":
            output_final = torch.stack(outputs).mean(dim=0)
            return output_final.float().cpu().numpy()

        output_stack = torch.stack(outputs)
        logits_stack = output_stack[:, :, : self.n_classes].float()
        stacked_mean = logits_stack.mean(dim=0)
        logits = self._apply_classification_temperature(stacked_mean)
        logits_np = logits.detach().cpu().numpy()

        if return_logits:
            return logits_np

        probs = self._classification_logits_to_probabilities(logits_stack).cpu().numpy()
        if self.version_config.enable_threshold_tuning:
            probs = self.postprocessor.apply_threshold(probs)
        return probs

    def predict_raw_logits(self, X_test: np.ndarray | torch.Tensor) -> np.ndarray:
        if self.model_type != "clf":
            raise RuntimeError(
                "predict_raw_logits is only available for classification."
            )
        if self.ensemble_results is None:
            raise RuntimeError(
                "Model must be fitted before prediction. Call fit() first."
            )
        if len(self._member_models) != len(self.ensemble_results):
            raise RuntimeError(
                "Cached ensemble member models are missing or stale. Call fit() again."
            )

        X_test = self._clean_predict_inputs(X_test)
        outputs: list[torch.Tensor] = []
        for member_model, augmentor_pipeline, cfg, (
            X_train_proc,
            y_train_proc,
            cat_ix_proc,
        ) in zip(
            self._member_models,
            self._get_augmentor_pipelines(),
            self._get_ensemble_configs(),
            self.ensemble_results,
        ):
            member_batches = []
            for start in range(0, len(X_test), self.inference_batch_size):
                end = min(len(X_test), start + self.inference_batch_size)
                member_batches.append(
                    self._predict_single_member(
                        cfg,
                        augmentor_pipeline,
                        member_model,
                        X_train_proc,
                        y_train_proc,
                        cat_ix_proc,
                        X_test[start:end],
                    )
                )
            outputs.append(torch.cat(member_batches, dim=0)[:, : self.n_classes])

        return torch.stack(outputs).float().cpu().numpy()

    def predict_debug_outputs(
        self, X_test: np.ndarray | torch.Tensor, *, member_index: int = 0
    ) -> dict[str, np.ndarray]:
        if self.model_type != "clf":
            raise RuntimeError(
                "predict_debug_outputs is only available for classification."
            )
        if self.ensemble_results is None or len(self._member_models) != len(
            self.ensemble_results
        ):
            raise RuntimeError("Model must be fitted before debug prediction.")
        if member_index < 0 or member_index >= len(self.ensemble_results):
            raise IndexError(f"Invalid member_index: {member_index}")

        X_test = self._clean_predict_inputs(X_test)
        augmentor_pipeline = self._get_augmentor_pipelines()[member_index]
        cfg = self._get_ensemble_configs()[member_index]
        (
            X_train_proc,
            y_train_proc,
            cat_ix_proc,
        ) = self.ensemble_results[member_index]
        member_model = self._member_models[member_index]
        debug = self._predict_single_member(
            cfg,
            augmentor_pipeline,
            member_model,
            X_train_proc,
            y_train_proc,
            cat_ix_proc,
            X_test,
            return_debug=True,
        )
        result: dict[str, np.ndarray] = {}
        for key, value in debug.items():
            if isinstance(value, torch.Tensor):
                result[key] = value.detach().float().cpu().numpy()
        return result

    def predict_proba(self, X_test: np.ndarray | torch.Tensor) -> np.ndarray:
        if self.model_type != "clf":
            raise RuntimeError("predict_proba is only available for classification.")
        return self.forward(X_test)

    def predict(self, X_test: np.ndarray | torch.Tensor) -> np.ndarray:
        out = self.forward(X_test)
        if self.model_type == "reg":
            return out
        return np.argmax(out, axis=1)

    def _fit_temperature_scaling(
        self, X_train: np.ndarray, y_train: np.ndarray, cat_ix: list[int]
    ) -> None:
        if len(X_train) < 10:
            return
        split = max(1, int(0.8 * len(X_train)))
        if split >= len(X_train):
            return
        X_fit, X_val = X_train[:split], X_train[split:]
        y_fit, y_val = y_train[:split], y_train[split:]
        clone = TabPFN(
            model_dir=self.model_dir,
            model_type=self.model_type,
            model_id=self.model_id,
            static_seed=self.static_seed,
            n_estimators=self.n_estimators,
            subsample_size=self.subsample_size,
            add_fingerprint_feature=self.add_fingerprint_feature,
            feature_shift_decoder=self.feature_shift_decoder,
            polynomial_features=self.polynomial_features,
            class_shift_method=self.class_shift_method,
            softmax_temperature=1.0,
            average_before_softmax=self.average_before_softmax,
            n_workers=self.n_workers,
            parallel_mode=self.parallel_mode,
            version=self.version_config.model_version,
            preprocessing_recipe=self.version_config.preprocessing_recipe,
            use_robust_scaling=self.version_config.use_robust_scaling,
            use_soft_clipping=self.version_config.use_soft_clipping,
            use_quantile_transform=self.version_config.use_quantile_transform,
            use_standard_scaling=self.version_config.use_standard_scaling,
            enable_temperature_scaling=False,
            enable_threshold_tuning=False,
            feature_group_size=self.version_config.feature_group_size,
            n_layers_cls=self.version_config.n_layers_cls,
            n_layers_reg=self.version_config.n_layers_reg,
            use_regression_mlp_encoder=self.version_config.use_regression_mlp_encoder,
            regression_mlp_hidden_dim=self.version_config.regression_mlp_hidden_dim,
            use_thinking_rows=self.version_config.use_thinking_rows,
            n_thinking_rows=self.version_config.n_thinking_rows,
            enable_flash_attention=self.version_config.enable_flash_attention,
            inference_batch_size=self.inference_batch_size,
            strict_version_match=self.strict_version_match,
        )
        clone.fit(X_fit, y_fit, cat_ix=cat_ix, random_state=self.static_seed)
        logits = clone.forward(X_val, return_logits=True)
        self.postprocessor.fit_temperature(logits, y_val)

    def _fit_threshold_tuning(
        self, X_train: np.ndarray, y_train: np.ndarray, cat_ix: list[int]
    ) -> None:
        if len(np.unique(y_train)) != 2:
            warnings.warn(
                "Threshold tuning supports binary classification only.",
                stacklevel=2,
            )
            return
        if len(X_train) < 10:
            return
        split = max(1, int(0.8 * len(X_train)))
        if split >= len(X_train):
            return
        X_fit, X_val = X_train[:split], X_train[split:]
        y_fit, y_val = y_train[:split], y_train[split:]
        clone = TabPFN(
            model_dir=self.model_dir,
            model_type=self.model_type,
            model_id=self.model_id,
            version=self.version_config.model_version,
            strict_version_match=self.strict_version_match,
        )
        clone.fit(X_fit, y_fit, cat_ix=cat_ix, random_state=self.static_seed)
        probs = clone.predict_proba(X_val)
        self.postprocessor.fit_threshold(probs[:, 1], y_val)

    def export_as_mlp(
        self,
        X: np.ndarray,
        *,
        hidden_dim: int = 128,
        n_steps: int = 50,
        lr: float = 1e-3,
    ) -> ExportArtifacts:
        if self.model_type == "clf":
            soft_targets = self.predict_proba(X)
        else:
            soft_targets = self.predict(X).reshape(-1, 1)
        return export_as_mlp_stub(
            X=np.asarray(X),
            soft_targets=np.asarray(soft_targets),
            hidden_dim=hidden_dim,
            n_steps=n_steps,
            lr=lr,
        )

    def export_as_tree_ensemble(self, *_: Any, **__: Any) -> ExportArtifacts:
        return export_as_tree_ensemble_stub()

    def save(self, path: str) -> None:
        payload = {
            "checkpoint_version": self.version_config.checkpoint_version,
            "model_version": self.version_config.model_version,
            "wrapper_version_config": self.version_config.to_dict(),
            "model_state_dict": self.model.state_dict(),
            "tabpfn_config": self.config,
            "postprocessor_state": self.postprocessor.state_dict(),
            "ensemble_augmentor": self.ensemble_augmentor,
            "ensemble_results": (
                list(self.ensemble_results)
                if self.ensemble_results is not None
                else None
            ),
            "member_models": self._member_models,
            "fit_cat_ix": self._fit_cat_ix,
            "ordinal_encoder": self._ordinal_encoder,
            "n_classes": self.n_classes,
            "model_type": self.model_type,
            "model_id": self.model_id,
            "static_seed": self.static_seed,
            "model_dir": self.model_dir,
            "softmax_temperature": self.softmax_temperature,
            "average_before_softmax": self.average_before_softmax,
            "strict_version_match": self.strict_version_match,
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str) -> "TabPFN":
        data = torch.load(path, map_location="cpu", weights_only=False)
        cfg = data.get("wrapper_version_config", {})
        init_kwargs = dict(
            model_dir=data["model_dir"],
            model_type=data["model_type"],
            model_id=int(data.get("model_id", 0)),
            static_seed=int(data.get("static_seed", 0)),
            softmax_temperature=float(data.get("softmax_temperature", 0.9)),
            average_before_softmax=bool(data.get("average_before_softmax", False)),
            strict_version_match=bool(data.get("strict_version_match", True)),
            version=cfg.get("model_version", "v2_6"),
            n_layers_cls=cfg.get("n_layers_cls"),
            n_layers_reg=cfg.get("n_layers_reg"),
            feature_group_size=cfg.get("feature_group_size"),
            use_regression_mlp_encoder=cfg.get("use_regression_mlp_encoder"),
            regression_mlp_hidden_dim=cfg.get("regression_mlp_hidden_dim"),
            use_thinking_rows=cfg.get("use_thinking_rows"),
            n_thinking_rows=cfg.get("n_thinking_rows"),
            preprocessing_recipe=cfg.get("preprocessing_recipe"),
            use_robust_scaling=cfg.get("use_robust_scaling", False),
            use_soft_clipping=cfg.get("use_soft_clipping", False),
            use_quantile_transform=cfg.get("use_quantile_transform", False),
            use_standard_scaling=cfg.get("use_standard_scaling", False),
            enable_temperature_scaling=cfg.get("enable_temperature_scaling", False),
            enable_threshold_tuning=cfg.get("enable_threshold_tuning", False),
            export_mode=cfg.get("export_mode", "icl"),
            enable_flash_attention=cfg.get("enable_flash_attention", False),
        )
        if cls is not TabPFN:
            init_kwargs.pop("model_type", None)
        model = cls(**init_kwargs)
        source_model_version = data.get(
            "model_version",
            cfg.get("model_version", model.version_config.model_version),
        )
        source_checkpoint_version = data.get(
            "checkpoint_version",
            source_model_version,
        )
        load_checkpoint_compatible(
            model=model.model,
            state_dict=data["model_state_dict"],
            source_checkpoint_version=str(source_checkpoint_version),
            source_model_version=str(source_model_version),
            target_model_version=model.version_config.model_version,
        )
        model.ensemble_augmentor = data.get("ensemble_augmentor")
        ensemble_results = data.get(
            "ensemble_results",
            data.get("ensemble_members", data.get("augmentors")),
        )
        if (
            ensemble_results is not None
            and len(ensemble_results) > 0
            and len(ensemble_results[0]) == 5
        ):
            configs = [member[0] for member in ensemble_results]
            pipelines = [member[1] for member in ensemble_results]
            ensemble_results = [
                (member[2], member[3], member[4])
                for member in ensemble_results
            ]
            if model.ensemble_augmentor is None:
                model.ensemble_augmentor = TabPFNEnsembleAugmentor(
                    task=(
                        "classification"
                        if model.model_type == "clf"
                        else "regression"
                    )
                )
            model.ensemble_augmentor.configs = configs
            model.ensemble_augmentor.augmentor_pipelines = pipelines
        elif (
            ensemble_results is not None
            and len(ensemble_results) > 0
            and len(ensemble_results[0]) == 4
        ):
            configs = [member[0] for member in ensemble_results]
            ensemble_results = [
                (member[1], member[2], member[3])
                for member in ensemble_results
            ]
            if model.ensemble_augmentor is not None:
                model.ensemble_augmentor.configs = configs
        model.ensemble_results = ensemble_results
        model._member_models = data.get("member_models", [])
        model._fit_cat_ix = list(data.get("fit_cat_ix", []))
        model._ordinal_encoder = data.get("ordinal_encoder")
        model.n_classes = data.get("n_classes")
        model.postprocessor.load_state_dict(data.get("postprocessor_state", {}))
        return model
