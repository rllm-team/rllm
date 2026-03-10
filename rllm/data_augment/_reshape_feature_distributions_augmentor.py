#  Copyright (c) Prior Labs GmbH 2025.

"""Reshape feature distributions using different transformations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from pandas.core.common import contextlib
from scipy.stats import shapiro
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)
from typing_extensions import override

from rllm.data_augment.data_augmentor import (
    DataAugmentor,
)
from rllm.data_augment.kdi_transformer_with_nan import (
    KDITransformerWithNaN,
    get_all_kdi_transformers,
)
from rllm.data_augment.none_transformer import NoneTransformer
from rllm.data_augment.utils import (
    _exp_minus_1,
    _identity,
    add_safe_standard_to_safe_power_without_standard,
    infer_random_state,
    make_box_cox_safe,
    make_standard_scaler_safe,
    skew,
)
from rllm.data_augment.safe_power_transformer import SafePowerTransformer

if TYPE_CHECKING:
    from sklearn.base import TransformerMixin


class ReshapeFeatureDistributionsAugmentor(DataAugmentor):
    """Reshape the feature distributions using different transformations."""

    @staticmethod
    def get_column_types(X: np.ndarray) -> list[str]:
        """Returns a list of column types for the given data, that indicate how
        the data should be preprocessed.
        """
        # TODO(eddiebergman): Bad to keep calling skew again and again here...
        column_types = []
        for col in range(X.shape[1]):
            if np.unique(X[:, col]).size < 10:
                column_types.append(f"ordinal_{col}")
            elif (
                skew(X[:, col]) > 1.1
                and np.min(X[:, col]) >= 0
                and np.max(X[:, col]) <= 1
            ):
                column_types.append(f"skewed_pos_1_0_{col}")
            elif skew(X[:, col]) > 1.1 and np.min(X[:, col]) > 0:
                column_types.append(f"skewed_pos_{col}")
            elif skew(X[:, col]) > 1.1:
                column_types.append(f"skewed_{col}")
            elif shapiro(X[0:3000, col]).statistic > 0.95:
                column_types.append(f"normal_{col}")
            else:
                column_types.append(f"other_{col}")
        return column_types

    @staticmethod
    def get_adaptive_augmentors(
        num_examples: int = 100,
        random_state: int | None = None,
    ) -> dict[str, ColumnTransformer]:
        """Returns a dictionary of adaptive column transformers that can be used to
        augment the data. Adaptive column transformers are used to augment the
        data based on the column type, they receive a pandas dataframe with column
        names, that indicate the column type. Column types are not datatypes,
        but rather a string that indicates how the data should be augmented.

        Args:
            num_examples: The number of examples in the dataset.
            random_state: The random state to use for the transformers.
        """
        return {
            "adaptive": ColumnTransformer(
                [
                    (
                        "skewed_pos_1_0",
                        FunctionTransformer(
                            func=np.exp,
                            inverse_func=np.log,
                            check_inverse=False,
                        ),
                        make_column_selector("skewed_pos_1_0*"),
                    ),
                    (
                        "skewed_pos",
                        make_box_cox_safe(
                            add_safe_standard_to_safe_power_without_standard(
                                SafePowerTransformer(
                                    standardize=False,
                                    method="box-cox",
                                ),
                            ),
                        ),
                        make_column_selector("skewed_pos*"),
                    ),
                    (
                        "skewed",
                        add_safe_standard_to_safe_power_without_standard(
                            SafePowerTransformer(
                                standardize=False,
                                method="yeo-johnson",
                            ),
                        ),
                        make_column_selector("skewed*"),
                    ),
                    (
                        "other",
                        QuantileTransformer(
                            output_distribution="normal",
                            n_quantiles=num_examples // 10,
                            random_state=random_state,
                        ),
                        # "other" or "ordinal"
                        make_column_selector("other*"),
                    ),
                    (
                        "ordinal",
                        NoneTransformer(),
                        # "other" or "ordinal"
                        make_column_selector("ordinal*"),
                    ),
                    (
                        "normal",
                        NoneTransformer(),
                        make_column_selector("normal*"),
                    ),
                ],
                remainder="passthrough",
            ),
        }

    @staticmethod
    def get_all_augmentors(
        num_examples: int,
        random_state: int | None = None,
    ) -> dict[str, TransformerMixin | Pipeline]:
        all_augmentors = {
            "power": add_safe_standard_to_safe_power_without_standard(
                PowerTransformer(standardize=False),
            ),
            "safepower": add_safe_standard_to_safe_power_without_standard(
                SafePowerTransformer(standardize=False),
            ),
            "power_box": make_box_cox_safe(
                add_safe_standard_to_safe_power_without_standard(
                    PowerTransformer(standardize=False, method="box-cox"),
                ),
            ),
            "safepower_box": make_box_cox_safe(
                add_safe_standard_to_safe_power_without_standard(
                    SafePowerTransformer(standardize=False, method="box-cox"),
                ),
            ),
            "log": FunctionTransformer(
                func=np.log,
                inverse_func=np.exp,
                check_inverse=False,
            ),
            "1_plus_log": FunctionTransformer(
                func=np.log1p,
                inverse_func=_exp_minus_1,
                check_inverse=False,
            ),
            "exp": FunctionTransformer(
                func=np.exp,
                inverse_func=np.log,
                check_inverse=False,
            ),
            "quantile_uni_coarse": QuantileTransformer(
                output_distribution="uniform",
                n_quantiles=max(num_examples // 10, 2),
                random_state=random_state,
            ),
            "quantile_norm_coarse": QuantileTransformer(
                output_distribution="normal",
                n_quantiles=max(num_examples // 10, 2),
                random_state=random_state,
            ),
            "quantile_uni": QuantileTransformer(
                output_distribution="uniform",
                n_quantiles=max(num_examples // 5, 2),
                random_state=random_state,
            ),
            "quantile_norm": QuantileTransformer(
                output_distribution="normal",
                n_quantiles=max(num_examples // 5, 2),
                random_state=random_state,
            ),
            "quantile_uni_fine": QuantileTransformer(
                output_distribution="uniform",
                n_quantiles=num_examples,
                random_state=random_state,
            ),
            "quantile_norm_fine": QuantileTransformer(
                output_distribution="normal",
                n_quantiles=num_examples,
                random_state=random_state,
            ),
            "robust": RobustScaler(unit_variance=True),
            "none": FunctionTransformer(_identity),
            **get_all_kdi_transformers(),
        }

        with contextlib.suppress(Exception):
            all_augmentors["norm_and_kdi"] = FeatureUnion(
                [
                    (
                        "norm",
                        QuantileTransformer(
                            output_distribution="normal",
                            n_quantiles=max(num_examples // 10, 2),
                            random_state=random_state,
                        ),
                    ),
                    (
                        "kdi",
                        KDITransformerWithNaN(alpha=1.0, output_distribution="uniform"),
                    ),
                ],
            )

        all_augmentors.update(
            ReshapeFeatureDistributionsAugmentor.get_adaptive_augmentors(
                num_examples,
                random_state=random_state,
            ),
        )

        return all_augmentors

    def get_all_global_transformers(
        self,
        num_examples: int,
        num_features: int,
        random_state: int | None = None,
    ) -> dict[str, FeatureUnion | Pipeline]:
        return {
            "scaler": make_standard_scaler_safe(("standard", StandardScaler())),
            "svd": FeatureUnion(
                [
                    ("passthrough", FunctionTransformer(func=_identity)),
                    (
                        "svd",
                        Pipeline(
                            steps=[
                                (
                                    "save_standard",
                                    make_standard_scaler_safe(
                                        ("standard", StandardScaler(with_mean=False)),
                                    ),
                                ),
                                (
                                    "svd",
                                    TruncatedSVD(
                                        algorithm="arpack",
                                        n_components=max(
                                            1,
                                            min(
                                                num_examples // 10 + 1,
                                                num_features // 2,
                                            ),
                                        ),
                                        random_state=random_state,
                                    ),
                                ),
                            ],
                        ),
                    ),
                ],
            ),
        }

    def __init__(
        self,
        *,
        transform_name: str = "safepower",
        apply_to_categorical: bool = False,
        append_to_original: bool = False,
        subsample_features: float = -1,
        global_transformer_name: str | None = None,
        random_state: int | np.random.Generator | None = None,
    ):
        super().__init__()
        self.transform_name = transform_name
        self.apply_to_categorical = apply_to_categorical
        self.append_to_original = append_to_original
        self.random_state = random_state
        self.subsample_features = float(subsample_features)
        self.global_transformer_name = global_transformer_name
        self.transformer_: Pipeline | ColumnTransformer | None = None

    def _set_transformer_and_cat_ix(  # noqa: PLR0912
        self,
        n_samples: int,
        n_features: int,
        categorical_features: list[int],
    ) -> tuple[Pipeline | ColumnTransformer, list[int]]:
        if "adaptive" in self.transform_name:
            raise NotImplementedError("Adaptive preprocessing raw removed.")

        static_seed, rng = infer_random_state(self.random_state)

        if (
            self.global_transformer_name is not None
            and self.global_transformer_name != "None"
            and not (self.global_transformer_name == "svd" and n_features < 2)
        ):
            global_transformer_ = self.get_all_global_transformers(
                n_samples,
                n_features,
                random_state=static_seed,
            )[self.global_transformer_name]
        else:
            global_transformer_ = None

        all_augmentors = self.get_all_augmentors(
            n_samples,
            random_state=static_seed,
        )
        if self.subsample_features > 0:
            subsample_features = int(self.subsample_features * n_features) + 1
            # sampling more features than exist
            replace = subsample_features > n_features
            self.subsampled_features_ = rng.choice(
                list(range(n_features)),
                subsample_features,
                replace=replace,
            )
            categorical_features = [
                new_idx
                for new_idx, idx in enumerate(self.subsampled_features_)
                if idx in categorical_features
            ]
            n_features = subsample_features
        else:
            self.subsampled_features_ = np.arange(n_features)

        all_feats_ix = list(range(n_features))
        transformers = []

        numerical_ix = [i for i in range(n_features) if i not in categorical_features]

        # -------- Append to original ------
        # If we append to original, all the categorical indices are kept in place
        # as the first transform is a passthrough on the whole X as it is above
        if self.append_to_original and self.apply_to_categorical:
            trans_ixs = categorical_features + numerical_ix
            transformers.append(("original", "passthrough", all_feats_ix))
            cat_ix = categorical_features  # Exist as they are in original

        elif self.append_to_original and not self.apply_to_categorical:
            trans_ixs = numerical_ix
            # Includes the categoricals passed through
            transformers.append(("original", "passthrough", all_feats_ix))
            cat_ix = categorical_features  # Exist as they are in original

        # -------- Don't append to original ------
        # We only have categorical indices if we don't transform them
        # The first transformer will be a passthrough on the categorical indices
        # Making them the first
        elif not self.append_to_original and self.apply_to_categorical:
            trans_ixs = categorical_features + numerical_ix
            cat_ix = []  # We have none left, they've been transformed

        elif not self.append_to_original and not self.apply_to_categorical:
            trans_ixs = numerical_ix
            transformers.append(("cats", "passthrough", categorical_features))
            cat_ix = list(range(len(categorical_features)))  # They are at start

        else:
            raise ValueError(
                f"Unrecognized combination of {self.apply_to_categorical=}"
                f" and {self.append_to_original=}",
            )

        # NOTE: No need to keep track of categoricals here, already done above
        if self.transform_name != "per_feature":
            _transformer = all_augmentors[self.transform_name]
            transformers.append(("feat_transform", _transformer, trans_ixs))
        else:
            augmentors = list(all_augmentors.values())
            transformers.extend(
                [
                    (f"transformer_{i}", rng.choice(augmentors), [i])  # type: ignore
                    for i in trans_ixs
                ],
            )

        transformer = ColumnTransformer(
            transformers,
            remainder="drop",
            sparse_threshold=0.0,  # No sparse
        )

        # Apply a global transformer which accepts the entire dataset instead of
        # one column
        # NOTE: We assume global_transformer does not destroy the semantic meaning of
        # categorical_features_.
        if global_transformer_:
            transformer = Pipeline(
                [
                    ("preprocess", transformer),
                    ("global_transformer", global_transformer_),
                ],
            )

        self.transformer_ = transformer

        return transformer, cat_ix

    @override
    def _fit(self, X: np.ndarray, categorical_features: list[int]) -> list[int]:
        n_samples, n_features = X.shape
        transformer, cat_ix = self._set_transformer_and_cat_ix(
            n_samples,
            n_features,
            categorical_features,
        )
        transformer.fit(X[:, self.subsampled_features_])
        self.categorical_features_after_transform_ = cat_ix
        self.transformer_ = transformer
        return cat_ix

    @override
    def fit_transform(
        self,
        X: np.ndarray,
        categorical_features: list[int],
    ) -> tuple[np.ndarray, list[int]]:
        n_samples, n_features = X.shape
        transformer, cat_ix = self._set_transformer_and_cat_ix(
            n_samples,
            n_features,
            categorical_features,
        )
        Xt = transformer.fit_transform(X[:, self.subsampled_features_])
        self.categorical_features_after_transform_ = cat_ix
        self.transformer_ = transformer
        return (Xt, cat_ix)  # type: ignore

    @override
    def _transform(self, X: np.ndarray, *, is_test: bool = False) -> np.ndarray:
        assert self.transformer_ is not None, "You must call fit first"
        return self.transformer_.transform(X[:, self.subsampled_features_])  # type: ignore
