from __future__ import annotations

from typing import Optional, List, Literal

import numpy as np
import torch

from .tabpfn_v2.tabpfn.base import initialize_tabpfn_model
from rllm.data_augment.tabpfnv2_augment import prepare_classification_ensemble


class TabPFNv2(torch.nn.Module):
    """
    TabPFNv2 model wrapper for classification tasks.

    This class wraps the TabPFNv2 model and provides a PyTorch nn.Module interface
    for easy integration with PyTorch workflows.

    Args:
        model_dir (str): Directory containing the TabPFNv2 model checkpoint.
        model_type (str): Type of model, either 'clf' for classification or 'reg' for regression.
            Default: 'clf'.
        model_id (int): Model ID to load. Default: 0.
        static_seed (int): Random seed for reproducibility. Default: 0.
        n_estimators (int): Number of ensemble estimators. Default: 4.
        subsample_size (int): Subsample size for each estimator. Default: 10000.
        add_fingerprint_feature (bool): Whether to add fingerprint features. Default: True.
        feature_shift_decoder (str): Feature shift decoder strategy. Default: 'shuffle'.
        polynomial_features (str): Polynomial features strategy. Default: 'no'.
        class_shift_method (str): Class shift method. Default: 'shuffle'.
        softmax_temperature (float): Temperature for softmax scaling. Default: 0.9.
        average_before_softmax (bool): Whether to average logits before softmax. Default: False.
        n_workers (int): Number of parallel workers for preprocessing. Default: 1.
        parallel_mode (str): Parallel processing mode. Default: 'block'.
    """

    def __init__(
        self,
        model_dir: str,
        model_type: Literal["clf", "reg"] = "clf",
        model_id: int = 0,
        static_seed: int = 0,
        n_estimators: int = 4,
        subsample_size: int = 10000,
        add_fingerprint_feature: bool = True,
        feature_shift_decoder: str = "shuffle",
        polynomial_features: str = "no",
        class_shift_method: str = "shuffle",
        softmax_temperature: float = 0.9,
        average_before_softmax: bool = False,
        n_workers: int = 1,
        parallel_mode: str = "block",
    ):
        super().__init__()

        # Store hyperparameters
        self.model_dir = model_dir
        self.model_type = model_type
        self.model_id = model_id
        self.static_seed = static_seed
        self.n_estimators = n_estimators
        self.subsample_size = subsample_size
        self.add_fingerprint_feature = add_fingerprint_feature
        self.feature_shift_decoder = feature_shift_decoder
        self.polynomial_features = polynomial_features
        self.class_shift_method = class_shift_method
        self.softmax_temperature = softmax_temperature
        self.average_before_softmax = average_before_softmax
        self.n_workers = n_workers
        self.parallel_mode = parallel_mode

        # Initialize model
        self.model, self.config = initialize_tabpfn_model(
            model_dir=model_dir,
            model_type=model_type,
            model_id=model_id,
            static_seed=static_seed,
        )

        # Ensemble configurations (will be set during fit)
        self.augmentors = None
        self.n_classes = None

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
        if self.model_type == "clf":
            y_train = y_train.astype(np.int64, copy=False)

        # Determine number of classes
        self.n_classes = len(np.unique(y_train))

        # Prepare classification ensemble
        self.augmentors = prepare_classification_ensemble(
            X_train=X_train,
            y_train=y_train,
            n_estimators=self.n_estimators,
            subsample_size=min(self.subsample_size, len(X_train)),
            add_fingerprint_feature=self.add_fingerprint_feature,
            feature_shift_decoder=self.feature_shift_decoder,
            polynomial_features=self.polynomial_features,
            class_shift_method=self.class_shift_method,
            random_state=random_state,
            cat_ix=cat_ix,
            n_workers=self.n_workers,
            parallel_mode=self.parallel_mode,
        )

    def forward(
        self,
        X_test: np.ndarray | torch.Tensor,
    ) -> np.ndarray:
        """
        Forward pass through the TabPFNv2 model with ensemble inference.

        Args:
            X_test: Test features of shape (n_test, n_features).

        Returns:
            Predicted probabilities of shape (n_test, n_classes).
        """
        if self.augmentors is None:
            raise RuntimeError(
                "Model must be fitted before prediction. Call fit() first."
            )

        if isinstance(X_test, torch.Tensor):
            X_test = X_test.detach().cpu().numpy()

        device = next(self.model.parameters()).device
        outputs = []

        # Iterate through all preprocessing configurations
        for (
            cfg,
            augmentor,
            X_train_proc,
            y_train_proc,
            cat_ix_proc,
        ) in self.augmentors:
            # Preprocess test data
            X_test_proc, _ = augmentor.transform(X_test)

            # Convert to tensors
            X_train_tensor = torch.as_tensor(
                X_train_proc, dtype=torch.float32, device=device
            )
            X_test_tensor = torch.as_tensor(
                X_test_proc, dtype=torch.float32, device=device
            )
            y_train_tensor = torch.as_tensor(
                y_train_proc, dtype=torch.float32, device=device
            )

            # Concatenate train and test, add batch dimension
            X_full = torch.cat([X_train_tensor, X_test_tensor], dim=0).unsqueeze(1)

            # Direct forward pass
            with torch.inference_mode():
                output = self.model(
                    None,  # style
                    X_full,
                    y_train_tensor,
                    only_return_standard_out=True,
                    categorical_inds=cat_ix_proc,
                    single_eval_pos=len(y_train_tensor),
                )

            # Remove batch dimension
            output = output.squeeze(1)

            # Apply temperature scaling
            if self.softmax_temperature != 1 and self.n_classes is not None:
                output = output[:, : self.n_classes].float() / self.softmax_temperature

            # Reverse class permutation if exists
            if cfg.class_permutation is not None:
                output = output[..., cfg.class_permutation]

            outputs.append(output)

        # Aggregate outputs
        if self.average_before_softmax:
            output_final = torch.stack(outputs).mean(dim=0)
            output_final = torch.nn.functional.softmax(output_final, dim=1)
        else:
            # Softmax each output before averaging
            outputs = [torch.nn.functional.softmax(o, dim=1) for o in outputs]
            output_final = torch.stack(outputs).mean(dim=0)

        # Convert to numpy
        prediction_probabilities = output_final.float().cpu().numpy()

        # Normalize to ensure probabilities sum to 1
        prediction_probabilities = (
            prediction_probabilities
            / prediction_probabilities.sum(axis=1, keepdims=True)
        )

        return prediction_probabilities
