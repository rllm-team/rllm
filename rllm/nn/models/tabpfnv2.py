from __future__ import annotations

from typing import Optional, List, Literal, Any, TYPE_CHECKING
import numpy as np
import os

import torch

from rllm.data_augment import (
    prepare_classification_ensemble,
    prepare_regression_ensemble,
)
from rllm.nn.models.tabpfn_v2.loading import load_model
from rllm.utils import download_model_from_huggingface

if TYPE_CHECKING:
    from rllm.nn.models.tabpfn_v2.config import InferenceConfig
    from rllm.nn.models.tabpfn_v2.tabpfn_backbone import (
        PerFeatureTransformer,
    )


def get_filename_from_model_name(
    model_type: str,
    model_id: Optional[int] = None,
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
        if model_id is None or model_id not in range(len(classifier_filenames)):
            return classifier_filenames[0]
        else:
            return classifier_filenames[model_id]
    elif model_type == "reg":
        if model_id is None or model_id not in range(len(regressor_filenames)):
            return regressor_filenames[0]
        else:
            return regressor_filenames[model_id]
    else:
        raise ValueError(f"Invalid model_type: {model_type}")


def load_model_criterion_config(
    model_dir: str,
    model_type: str,
    model_id: Optional[int] = None,
    *,
    check_bar_distribution_criterion: bool,
    cache_trainset_representation: bool,
    model_seed: int,
) -> tuple[PerFeatureTransformer, InferenceConfig, Any]:
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

    Returns:
        The model, criterion, and config.
    """

    model_name = get_filename_from_model_name(model_type, model_id)
    model_path = os.path.join(model_dir, model_name)

    print(f"Loading model from {model_dir}...")
    if not os.path.exists(model_path):
        download_model_from_huggingface(
            repo=f"Prior-Labs/TabPFN-v2-{model_type}",
            model_name=model_name,
            download_path=model_dir,
        )
    loaded_model, config, criterion = load_model(
        path=model_path,
        model_seed=model_seed,
    )
    loaded_model.cache_trainset_representation = cache_trainset_representation
    return loaded_model, config, criterion


def initialize_tabpfn_model(
    model_dir: str,
    model_type: str,
    model_id: int,
    static_seed: int,
) -> tuple[PerFeatureTransformer, InferenceConfig, object]:
    """Common logic to load the TabPFN model, set up the random state,
    and optionally download the model.

    Args:
        model_path: Path or directive ("auto") to load the pre-trained model from.
        which: Which TabPFN model to load.
        fit_mode: Determines caching behavior.
        static_seed: Random seed for reproducibility logic.

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
        )

    return model, config_, criterion


class TabPFNv2(torch.nn.Module):
    """
    TabPFNv2 model wrapper for classification and regression tasks.

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
        self.model, self.config, self.bardist = initialize_tabpfn_model(
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
            self.n_classes = len(np.unique(y_train))
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
        else:
            y_train = y_train.astype(np.float32, copy=False)
            self.augmentors = prepare_regression_ensemble(
                X_train=X_train,
                y_train=y_train,
                n_estimators=self.n_estimators,
                subsample_size=min(self.subsample_size, len(X_train)),
                add_fingerprint_feature=self.add_fingerprint_feature,
                feature_shift_decoder=self.feature_shift_decoder,
                polynomial_features=self.polynomial_features,
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

        For classification (model_type='clf'), returns predicted probabilities
        of shape (n_test, n_classes).

        For regression (model_type='reg'), returns predicted values
        of shape (n_test,).

        Args:
            X_test: Test features of shape (n_test, n_features).

        Returns:
            Predictions array.
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

            if self.model_type == "reg":
                # For regression, apply inverse target transform if present
                if cfg.target_transform is not None:
                    mean_pred = self.bardist.mean(output).float()
                    mean_np = mean_pred.cpu().numpy().reshape(-1, 1)
                    mean_np = cfg.target_transform.inverse_transform(mean_np).ravel()
                    outputs.append(torch.as_tensor(mean_np, device=device))
                else:
                    outputs.append(self.bardist.mean(output).float())
            else:
                # Apply temperature scaling for classification
                if self.softmax_temperature != 1 and self.n_classes is not None:
                    output = (
                        output[:, : self.n_classes].float() / self.softmax_temperature
                    )

                # Reverse class permutation if exists
                if cfg.class_permutation is not None:
                    output = output[..., cfg.class_permutation]

                outputs.append(output)

        if self.model_type == "reg":
            output_final = torch.stack(outputs).mean(dim=0)
            return output_final.float().cpu().numpy()

        # Classification: aggregate with softmax
        if self.average_before_softmax:
            output_final = torch.stack(outputs).mean(dim=0)
            output_final = torch.nn.functional.softmax(output_final, dim=1)
        else:
            # Softmax each output before averaging
            outputs = [torch.nn.functional.softmax(o, dim=1) for o in outputs]
            output_final = torch.stack(outputs).mean(dim=0)

        # Normalize to ensure probabilities sum to 1
        prediction_probabilities = output_final / output_final.sum(
            axis=1, keepdims=True
        )

        return prediction_probabilities
