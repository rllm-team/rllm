from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
from typing import Literal, Any
import numpy as np
import warnings

import torch

from rllm.data_augment import TabPFNEnsembleAugmentor
from .tabpfn_internal.input_cleaning import (
    clean_data as clean_tabpfn_data,
    fix_dtypes as fix_tabpfn_dtypes,
    process_text_na_dataframe,
)
from .tabpfn_internal.loading import initialize_tabpfn_model
from .tabpfn_internal.regression import translate_probs_across_borders
from rllm.types import ColType


ArrayLike = np.ndarray | torch.Tensor
ModelVersion = Literal["v2_6"]


def _to_numpy(x: ArrayLike) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


class TabPFN(torch.nn.Module):
    """TabPFN checkpoint wrapper with PyTorch and sklearn-style inference APIs."""

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
            static_seed=static_seed,
            metadata=metadata,
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
        """Prepare ensemble preprocessing and train-context caches for inference."""
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
        cat_ix_proc: list[int],
        X_test: np.ndarray,
    ) -> torch.Tensor:
        device = next(self.model.parameters()).device
        X_test_proc, _ = augmentor_pipeline.transform(X_test)
        X_test_tensor = torch.as_tensor(X_test_proc, dtype=torch.float32, device=device)
        self.model.to(device)

        X_train_tensor = torch.as_tensor(
            X_train_proc, dtype=torch.float32, device=device
        )
        y_train_tensor = torch.as_tensor(
            y_train_proc, dtype=torch.float32, device=device
        )
        X_full = torch.cat([X_train_tensor, X_test_tensor], dim=0).unsqueeze(1)
        y_train_tensor = y_train_tensor.unsqueeze(1)
        with torch.inference_mode(), self._sdp_context():
            output = self.model(
                X_full,
                y_train_tensor,
                only_return_standard_out=True,
                categorical_inds=cat_ix_proc,
                single_eval_pos=len(y_train_proc),
            )
        output = output.squeeze(1)
        if self.model_type == "reg":
            if self.softmax_temperature != 1:
                output = output / self.softmax_temperature
            if cfg.target_transform is None:
                probs = translate_probs_across_borders(
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
        X_test: ArrayLike,
    ) -> np.ndarray:
        X_test = self._clean_predict_inputs(X_test)
        outputs: list[torch.Tensor] = []

        for (
            augmentor_pipeline,
            cfg,
            (X_train_proc, y_train_proc, cat_ix_proc),
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
                        cat_ix_proc,
                        X_test[start:end],
                    )
                )
            output = torch.cat(member_batches, dim=0)
            outputs.append(output)

        if self.model_type == "reg":
            output_final = torch.stack(outputs).mean(dim=0)
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
