from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
from typing import Literal, Any
import numpy as np
import warnings

import torch

from rllm.data_augment import TabPFNEnsembleAugmentor
from .tabpfn_runtime.input_cleaning import (
    clean_data as clean_tabpfn_data,
    fix_dtypes as fix_tabpfn_dtypes,
    process_text_na_dataframe,
)
from .tabpfn_runtime.loading import initialize_tabpfn_model
from .tabpfn_runtime.utils import translate_probs_across_borders
from .tabpfn_runtime.utils import transform_borders_one
from rllm.types import ColType


ArrayLike = np.ndarray | torch.Tensor
ModelVersion = Literal["v2_6"]
InferencePrecision = Literal["auto", "autocast"] | torch.dtype


def _to_numpy(x: ArrayLike) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _is_autocast_available(device: torch.device) -> bool:
    if device.type == "cpu":
        return False
    try:
        from torch.amp.autocast_mode import is_autocast_available

        return bool(is_autocast_available(device.type))
    except Exception:
        return False


class TabPFN(torch.nn.Module):
    """TabPFN model wrapper for classification and regression tasks.

    This class wraps the retained TabPFN checkpoint and provides a PyTorch
    ``nn.Module`` interface plus sklearn-style ``fit``/``predict`` helpers.
    Calling ``fit`` prepares TabPFN's train-context preprocessing and ensemble
    configurations; it does not update the checkpoint weights.

    Args:
        model_dir (str): Directory containing the TabPFN checkpoint files.
        model_type (str): Task type, either ``"clf"`` for classification or
            ``"reg"`` for regression. Default: ``"clf"``.
        model_id (int): Model ID to load. Default: ``0``.
        static_seed (int): Stored for API compatibility with earlier wrappers.
            Default: ``0``.
        n_estimators (int): Number of ensemble estimators. Default: ``4``.
        subsample_size (int, optional): Maximum number of training samples used
            by each estimator. If ``None``, each estimator can use all training
            samples. Default: ``None``.
        add_fingerprint_feature (bool): Whether to add fingerprint features.
            Default: ``True``.
        feature_shift_decoder (str): Feature shift decoder strategy. Default:
            ``"shuffle"``.
        polynomial_features (str or int): Polynomial feature strategy. For
            regression on v2.6, ``"no"`` is internally mapped to ``10`` to match
            the retained runtime. Default: ``"no"``.
        class_shift_method (str): Class shift strategy for classification
            ensembles. Default: ``"shuffle"``.
        softmax_temperature (float): Temperature for classification logits and
            regression bucket logits. Default: ``0.9``.
        average_before_softmax (bool): Whether to average classification logits
            before softmax. Default: ``False``.
        metadata (dict, optional): Reserved for compatibility with table-model
            construction APIs. Default: ``None``.
        version (str): Retained TabPFN checkpoint version. Default: ``"v2_6"``.
        enable_flash_attention (bool): Whether to allow PyTorch SDPA flash
            attention kernels during inference. Default: ``False``.
        inference_batch_size (int): Number of test rows evaluated per ensemble
            member at once. Default: ``4096``.
        inference_precision: Inference precision policy. Pass ``torch.float32``
            or ``torch.float64`` to force input/model arithmetic precision, ``"auto"``
            to enable PyTorch autocast on supported non-CPU devices, or
            ``"autocast"`` to require autocast support. Default: ``torch.float32``.
        strict_version_match (bool): Whether checkpoint/runtime version
            mismatches should raise an error. Default: ``True``.
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
        metadata: dict[ColType, list[dict[Any, Any]]] | None = None,
        version: ModelVersion = "v2_6",
        enable_flash_attention: bool = False,
        inference_batch_size: int = 4096,
        inference_precision: InferencePrecision = torch.float32,
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
        self.inference_batch_size = int(inference_batch_size)
        self.inference_precision = inference_precision
        self.use_autocast_ = False
        self.forced_inference_dtype_: torch.dtype | None = torch.float32
        self.strict_version_match = bool(strict_version_match)
        self.model_version = version
        self.enable_flash_attention = bool(enable_flash_attention)
        if polynomial_features == "no" and model_type == "reg" and version == "v2_6":
            polynomial_features = 10
        self.polynomial_features = polynomial_features

        self.subsample_size = None if subsample_size is None else int(subsample_size)
        self._fit_cat_ix: list[int] = []
        self._ordinal_encoder = None
        self.y_train_mean_ = 0.0
        self.y_train_std_ = 1.0
        self.znorm_space_bardist_ = None
        self.raw_space_bardist_ = None

        # Initialize model
        self.model, self.config, self.bardist = initialize_tabpfn_model(
            model_dir=model_dir,
            model_type=model_type,
            model_id=model_id,
            version=version,
            strict_version_match=self.strict_version_match,
        )
        if self.bardist is not None:
            self.znorm_space_bardist_ = deepcopy(self.bardist).float()
            self.raw_space_bardist_ = deepcopy(self.bardist).float()

        self.ensemble_augmentor: TabPFNEnsembleAugmentor | None = None
        # Fitted ensemble results produced by TabPFNEnsembleAugmentor.fit().
        self.ensemble_results = None
        self.n_classes = None

    def _resolve_fit_subsample_size(self, n_samples: int) -> int | None:
        if self.subsample_size is None:
            return None
        return min(int(self.subsample_size), int(n_samples))

    def _classification_logits_to_probabilities(
        self,
        raw_logits: torch.Tensor,
    ) -> torch.Tensor:
        if raw_logits.ndim < 2:
            raise ValueError(
                f"Expected logits with 2 or more dims, got {raw_logits.ndim}"
            )

        logits = raw_logits.float()
        if self.softmax_temperature != 1 and self.n_classes is not None:
            logits = logits / float(self.softmax_temperature)

        if logits.ndim >= 3:
            if self.average_before_softmax:
                probs = torch.nn.functional.softmax(logits.mean(dim=0), dim=-1)
            else:
                probs = torch.nn.functional.softmax(logits, dim=-1).mean(dim=0)
        else:
            probs = torch.nn.functional.softmax(logits, dim=-1)

        return probs / probs.sum(dim=-1, keepdim=True)

    def fit(
        self,
        X_train: ArrayLike,
        y_train: ArrayLike,
        cat_ix: list[int] | None = None,
        random_state: int = 0,
    ):
        """Prepare ensemble configurations for inference.

        Args:
            X_train: Training features of shape ``(n_samples, n_features)``.
            y_train: Training labels or regression targets of shape
                ``(n_samples,)``.
            cat_ix: Categorical feature indices. Default: ``None``.
            random_state: Random state for ensemble preprocessing. Default:
                ``0``.

        Returns:
            self: The fitted wrapper.
        """
        cat_ix = [] if cat_ix is None else list(cat_ix)
        X_train = _to_numpy(X_train)
        y_train = _to_numpy(y_train).reshape(-1)
        X_train, self._ordinal_encoder = clean_tabpfn_data(X_train, cat_ix)

        augmentor_kwargs = dict(
            n_estimators=self.n_estimators,
            subsample_size=self._resolve_fit_subsample_size(len(X_train)),
            add_fingerprint_feature=self.add_fingerprint_feature,
            feature_shift_decoder=self.feature_shift_decoder,
            polynomial_features=self.polynomial_features,
            random_state=random_state,
            max_index=len(X_train),
        )
        if self.model_type == "clf":
            y_train = y_train.astype(np.int64, copy=False)
            self.n_classes = len(np.unique(y_train))
            ensemble_augmentor = TabPFNEnsembleAugmentor(
                task="classification",
                class_shift_method=self.class_shift_method,
                n_classes=int(y_train.max()) + 1,
                **augmentor_kwargs,
            )
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
                task="regression",
                **augmentor_kwargs,
            )

        self.ensemble_results = list(
            ensemble_augmentor.fit(X_train=X_train, y_train=y_train, cat_ix=cat_ix)
        )
        self.ensemble_augmentor = ensemble_augmentor
        self._fit_cat_ix = list(cat_ix)
        return self

    def _clean_predict_inputs(self, X: ArrayLike) -> np.ndarray:
        X_df = fix_tabpfn_dtypes(_to_numpy(X), self._fit_cat_ix)
        return process_text_na_dataframe(
            X=X_df,
            ord_encoder=self._ordinal_encoder,
            fit_encoder=False,
        )

    def _sdp_context(self):
        if not self.enable_flash_attention:
            return nullcontext()
        if not torch.cuda.is_available():
            return nullcontext()
        try:
            from torch.nn.attention import SDPBackend, sdpa_kernel

            return sdpa_kernel(
                [
                    SDPBackend.FLASH_ATTENTION,
                    SDPBackend.EFFICIENT_ATTENTION,
                    SDPBackend.MATH,
                ]
            )
        except Exception:
            warnings.warn(
                "Flash attention kernel selection failed; falling back to "
                "default SDPA.",
                stacklevel=2,
            )
            return nullcontext()

    def _resolve_inference_precision(
        self,
        device: torch.device,
    ) -> tuple[bool, torch.dtype | None]:
        if self.inference_precision in ("auto", "autocast"):
            use_autocast = _is_autocast_available(device)
            if self.inference_precision == "autocast" and not use_autocast:
                raise ValueError(
                    "inference_precision='autocast' was requested, but PyTorch "
                    f"autocast is not available for device={device}."
                )
            self.use_autocast_ = use_autocast
            self.forced_inference_dtype_ = None
            return use_autocast, None

        if isinstance(self.inference_precision, torch.dtype):
            self.use_autocast_ = False
            self.forced_inference_dtype_ = self.inference_precision
            return False, self.inference_precision

        raise ValueError(
            "inference_precision must be one of {'auto', 'autocast'} or a torch.dtype."
        )

    def _autocast_context(self, device: torch.device):
        use_autocast, _forced_dtype = self._resolve_inference_precision(device)
        if device.type == "mps":
            return nullcontext()
        return torch.autocast(device.type, enabled=use_autocast)

    def _iter_ensemble_members(self):
        if self.ensemble_augmentor is None:
            raise RuntimeError(
                "Model must be fitted before prediction. Call fit() first."
            )
        pipelines = self.ensemble_augmentor.augmentor_pipelines
        configs = self.ensemble_augmentor.configs
        if self.ensemble_results is None:
            raise RuntimeError(
                "Model must be fitted before prediction. Call fit() first."
            )
        if len(pipelines) != len(self.ensemble_results) or len(configs) != len(
            self.ensemble_results
        ):
            raise RuntimeError(
                "Fitted ensemble configuration is missing or stale. Call fit() again."
            )
        return zip(pipelines, configs, self.ensemble_results)

    def _predict_single_member(
        self,
        cfg: Any,
        augmentor_pipeline: Any,
        X_train_proc: np.ndarray,
        y_train_proc: np.ndarray,
        X_test: np.ndarray,
    ) -> torch.Tensor:
        device = next(self.model.parameters()).device
        _use_autocast, forced_dtype = self._resolve_inference_precision(device)
        input_dtype = forced_dtype if forced_dtype is not None else torch.float32
        self.model.to(device=device, dtype=input_dtype)
        X_test_proc, _ = augmentor_pipeline.transform(X_test)
        X_test_tensor = torch.as_tensor(X_test_proc, dtype=input_dtype, device=device)

        X_train_tensor = torch.as_tensor(
            X_train_proc, dtype=input_dtype, device=device
        )
        y_train_tensor = torch.as_tensor(
            y_train_proc, dtype=input_dtype, device=device
        )
        X_full = torch.cat([X_train_tensor, X_test_tensor], dim=0).unsqueeze(1)
        y_train_tensor = y_train_tensor.unsqueeze(1)
        with torch.inference_mode(), self._sdp_context(), self._autocast_context(device):
            output = self.model(
                X_full,
                y_train_tensor,
                only_return_standard_out=True,
                single_eval_pos=len(y_train_proc),
            )
        output = output.squeeze(1)
        if self.model_type == "reg":
            output = output.float()
            if self.softmax_temperature != 1:
                output = output / self.softmax_temperature

            target_borders = self.znorm_space_bardist_.borders.to(output.device)
            if cfg.target_transform is not None:
                logit_cancel_mask, _descending_borders, borders_t = (
                    transform_borders_one(
                        self.znorm_space_bardist_.borders.cpu().numpy(),
                        target_transform=cfg.target_transform,
                        repair_nan_borders_after_transform=True,
                    )
                )
                if logit_cancel_mask is not None:
                    output = output.clone()
                    output[..., logit_cancel_mask] = float("-inf")
                source_borders = torch.as_tensor(
                    borders_t,
                    device=output.device,
                    dtype=target_borders.dtype,
                )
            else:
                source_borders = target_borders

            return translate_probs_across_borders(
                output,
                frm=source_borders,
                to=target_borders,
            )

        if cfg.class_permutation is not None:
            output = output[..., cfg.class_permutation]
        return output

    def forward(
        self,
        X_test: ArrayLike,
    ) -> np.ndarray:
        """Run ensemble inference with the fitted TabPFN train context.

        For classification, returns predicted probabilities with shape
        ``(n_test, n_classes)``. For regression, returns predicted values with
        shape ``(n_test,)``.

        Args:
            X_test: Test features of shape ``(n_test, n_features)``.

        Returns:
            A NumPy array containing probabilities or regression predictions.
        """
        X_test = self._clean_predict_inputs(X_test)
        outputs: list[torch.Tensor] = []

        for (
            augmentor_pipeline,
            cfg,
            (X_train_proc, y_train_proc, _cat_ix_proc),
        ) in self._iter_ensemble_members():
            member_batches = []
            for start in range(0, len(X_test), self.inference_batch_size):
                end = min(len(X_test), start + self.inference_batch_size)
                member_batches.append(
                    self._predict_single_member(
                        cfg,
                        augmentor_pipeline,
                        X_train_proc,
                        y_train_proc,
                        X_test[start:end],
                    )
                )
            output = torch.cat(member_batches, dim=0)
            outputs.append(output)

        if self.model_type == "reg":
            outputs_stack = torch.stack(outputs)
            if self.average_before_softmax:
                probs = outputs_stack.log().mean(dim=0).softmax(dim=-1)
            else:
                probs = outputs_stack.mean(dim=0)
            output_final = self.raw_space_bardist_.mean(probs.log())
            return output_final.float().cpu().numpy()

        logits_stack = torch.stack(outputs)[:, :, : self.n_classes].float()
        probs = self._classification_logits_to_probabilities(logits_stack).cpu().numpy()
        return probs

    def predict_proba(self, X_test: ArrayLike) -> np.ndarray:
        if self.model_type != "clf":
            raise RuntimeError("predict_proba is only available for classification.")
        return self.forward(X_test)

    def predict(self, X_test: ArrayLike) -> np.ndarray:
        out = self.forward(X_test)
        if self.model_type == "reg":
            return out
        return np.argmax(out, axis=1)
