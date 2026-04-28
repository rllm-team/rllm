from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, check_is_fitted
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder

_NA_PLACEHOLDER = "__rllm_tabpfn_na__"


class OrderPreservingColumnTransformer(ColumnTransformer):
    """ColumnTransformer variant that restores original column order."""

    def __init__(
        self,
        transformers: Sequence[
            tuple[
                str,
                BaseEstimator,
                str
                | int
                | slice
                | Iterable[str | int]
                | Callable[[Any], Iterable[str | int]],
            ]
        ],
        **kwargs: Any,
    ) -> None:
        super().__init__(transformers=transformers, **kwargs)
        assert all(
            isinstance(t, OneToOneFeatureMixin)
            for name, t, _ in transformers
            if name != "remainder"
        )
        assert len([t for name, _, t in transformers if name != "remainder"]) <= 1

    def transform(self, X: pd.DataFrame | np.ndarray, **kwargs: Any) -> np.ndarray:
        original_columns = X.columns if isinstance(X, pd.DataFrame) else range(X.shape[-1])
        X_t = super().transform(X, **kwargs)
        return self._preserve_order(X=X_t, original_columns=original_columns)

    def fit_transform(
        self,
        X: pd.DataFrame | np.ndarray,
        y: np.ndarray | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        original_columns = X.columns if isinstance(X, pd.DataFrame) else range(X.shape[-1])
        X_t = super().fit_transform(X, y, **kwargs)
        return self._preserve_order(X=X_t, original_columns=original_columns)

    def _preserve_order(
        self,
        X: np.ndarray,
        original_columns: list | range | pd.Index,
    ) -> np.ndarray:
        check_is_fitted(self)
        assert X.ndim == 2, f"Expected 2D input, got {X.ndim}D (shape={X.shape})"
        for name, _, col_subset in reversed(self.transformers_):
            if len(col_subset) > 0 and len(col_subset) < X.shape[-1] and name != "remainder":
                col_subset_list = list(col_subset)
                transformed_columns = col_subset_list + [
                    c for c in original_columns if c not in col_subset_list
                ]
                indices = [transformed_columns.index(c) for c in original_columns]
                X = X[:, indices]
        return X


def get_ordinal_encoder() -> OrderPreservingColumnTransformer:
    """Create the categorical encoder used by TabPFN preprocessing."""

    oe = OrdinalEncoder(
        categories="auto",
        dtype=np.float64,
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        encoded_missing_value=np.nan,
    )
    return OrderPreservingColumnTransformer(
        transformers=[("encoder", oe, make_column_selector(dtype_include=["category", "string"]))],
        remainder=FunctionTransformer(),
        sparse_threshold=0.0,
        verbose_feature_names_out=False,
    )


def fix_dtypes(
    X: pd.DataFrame | np.ndarray,
    cat_indices: Sequence[int] | None,
    numeric_dtype: str = "float64",
) -> pd.DataFrame:
    """Convert supported table inputs to a DataFrame with TabPFN dtypes."""

    if isinstance(X, pd.DataFrame):
        convert_dtype = True
    elif isinstance(X, np.ndarray):
        if X.dtype.kind in "?bBiufm":
            X = pd.DataFrame(X, copy=False, dtype=numeric_dtype)
            convert_dtype = False
        elif X.dtype.kind in "OV":
            X = pd.DataFrame(X, copy=True)
            convert_dtype = True
        elif X.dtype.kind in "SaU":
            raise ValueError(f"String dtypes are not supported. Got dtype: {X.dtype}")
        else:
            raise ValueError(f"Invalid dtype for X: {X.dtype}")
    else:
        raise ValueError(f"Invalid type for X: {type(X)}")

    if cat_indices is not None:
        is_numeric_indices = all(isinstance(i, (int, np.integer)) for i in cat_indices)
        columns_are_numeric = all(
            isinstance(col, (int, np.integer)) for col in X.columns.tolist()
        )
        use_col_names = is_numeric_indices and not columns_are_numeric
        if use_col_names:
            cat_col_names = [X.columns[i] for i in cat_indices]
            X[cat_col_names] = X[cat_col_names].astype("category")
        else:
            X[list(cat_indices)] = X[list(cat_indices)].astype("category")

    if convert_dtype:
        X = X.convert_dtypes()

    numerical_columns = X.select_dtypes(include=["number"]).columns
    if len(numerical_columns) > 0:
        X[numerical_columns] = X[numerical_columns].astype(numeric_dtype)
    return X


def process_text_na_dataframe(
    X: pd.DataFrame,
    placeholder: str = _NA_PLACEHOLDER,
    ord_encoder: ColumnTransformer | None = None,
    *,
    fit_encoder: bool = False,
) -> np.ndarray:
    """Encode text columns while preserving missing values as NaNs."""

    string_cols = X.select_dtypes(include=["string", "object"]).columns
    if len(string_cols) > 0:
        X[string_cols] = X[string_cols].fillna(placeholder)

    if fit_encoder and ord_encoder is not None:
        X_encoded = ord_encoder.fit_transform(X)
    elif ord_encoder is not None:
        X_encoded = ord_encoder.transform(X)
    else:
        X_encoded = X.to_numpy()

    string_cols_ix = [X.columns.get_loc(col) for col in string_cols]
    placeholder_mask = X[string_cols] == placeholder
    X_encoded[:, string_cols_ix] = np.where(
        placeholder_mask,
        np.nan,
        X_encoded[:, string_cols_ix],
    )
    return X_encoded.astype(np.float64)


def clean_data(
    X: np.ndarray,
    cat_indices: Sequence[int] | None,
) -> tuple[np.ndarray, ColumnTransformer]:
    """Clean training features and return the fitted ordinal encoder."""

    X_pandas = fix_dtypes(X=X, cat_indices=cat_indices)
    ord_encoder = get_ordinal_encoder()
    X_numpy = process_text_na_dataframe(
        X=X_pandas,
        ord_encoder=ord_encoder,
        fit_encoder=True,
    )
    return X_numpy, ord_encoder
