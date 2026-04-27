from dataclasses import dataclass
from typing import Literal, Union

from pandas import Series

from rllm.types import ColType


@dataclass
class FillNAConfig:
    """Configuration for missing-value imputation by column type.
    It centralizes fill strategies and fallback values for numerical,
    categorical, text, and timestamp columns. These options are consumed by
    :func:`fillna_by_coltype` and related helpers during preprocessing.

    Args:
        numerical_strategy (Literal["mean", "median", "mode", "constant"]):
            Strategy for numerical columns.
        numerical_fill_value (float): Constant fallback for numerical columns.
        categorical_fill_value (Union[int, str]): Fill value for categorical
            columns.
        text_fill_value (str): Fill value for text columns.
        timestamp_strategy (Literal["ffill", "bfill", "median", "constant"]):
            Strategy for timestamp columns.
        timestamp_fill_value: Constant fallback for timestamp columns when
            ``timestamp_strategy="constant"``.
    """

    numerical_strategy: Literal["mean", "median", "mode", "constant"] = "mean"
    numerical_fill_value: float = 0.0
    categorical_fill_value: Union[int, str] = -1
    text_fill_value: str = ""
    timestamp_strategy: Literal["ffill", "bfill", "median", "constant"] = "ffill"
    timestamp_fill_value: object = None


def fillna_numerical(
    col_series: Series,
    strategy: Literal["mean", "median", "mode", "constant"] = "mean",
    fill_value: float = 0.0,
) -> Series:
    """
    Fill missing values for numerical columns.

    Args:
        col_series: pandas Series with numerical data.
        strategy: Fill strategy. One of:
            - ``'mean'``: fill with column mean (default).
            - ``'median'``: fill with column median.
            - ``'mode'``: fill with most frequent value.
            - ``'constant'``: fill with ``fill_value``.
        fill_value: Fallback constant used when ``strategy='constant'`` or
            when all values are NaN. Defaults to ``0.0``.

    Returns:
        Series with missing values filled.

    Raises:
        ValueError: If ``strategy`` is not one of the accepted values.
    """
    if not col_series.isnull().any():
        return col_series.copy()

    if col_series.isnull().all():
        return col_series.fillna(fill_value)

    if strategy == "mean":
        value = col_series.mean()
    elif strategy == "median":
        value = col_series.median()
    elif strategy == "mode":
        mode_result = col_series.mode()
        value = float(mode_result.iloc[0]) if not mode_result.empty else fill_value
    elif strategy == "constant":
        value = fill_value
    else:
        raise ValueError(
            f"Unknown strategy '{strategy}'. "
            "Choose from 'mean', 'median', 'mode', 'constant'."
        )

    return col_series.fillna(value)


def fillna_categorical(
    col_series: Series,
    fill_value: Union[int, str] = -1,
) -> Series:
    """
    Fill missing values for categorical columns.

    Args:
        col_series: pandas Series with categorical (encoded) data.
        fill_value: Sentinel value used to represent missing/unknown category.
            Defaults to ``-1``.

    Returns:
        Series with missing values filled.
    """
    if not col_series.isnull().any():
        return col_series.copy()

    return col_series.fillna(fill_value)


def fillna_binary(col_series: Series) -> Series:
    """
    Fill missing values for binary columns with the mode (most frequent value).
    Falls back to ``0`` when the mode cannot be determined (all values are NaN).

    Args:
        col_series: pandas Series with binary data (0/1 or True/False).

    Returns:
        Series with missing values filled.
    """
    if not col_series.isnull().any():
        return col_series.copy()

    mode_result = col_series.mode()
    fill_value = mode_result.iloc[0] if not mode_result.empty else 0
    return col_series.fillna(fill_value)


def fillna_text(
    col_series: Series,
    fill_value: str = "",
) -> Series:
    """
    Fill missing values for text columns with an empty string.

    Args:
        col_series: pandas Series with text/string data.
        fill_value: String used to replace NaN values. Defaults to ``""``.

    Returns:
        Series with missing values filled.
    """
    if not col_series.isnull().any():
        return col_series.copy()

    return col_series.fillna(fill_value)


def fillna_timestamp(
    col_series: Series,
    strategy: Literal["ffill", "bfill", "median", "constant"] = "ffill",
    fill_value=None,
) -> Series:
    """
    Fill missing values for timestamp/datetime columns.

    Args:
        col_series: pandas Series with datetime data.
        strategy: Fill strategy. One of:
            - ``'ffill'``: forward-fill; remaining leading NaNs are back-filled
              (default).
            - ``'bfill'``: backward-fill; remaining trailing NaNs are
              forward-filled.
            - ``'median'``: fill with the median timestamp.
            - ``'constant'``: fill with ``fill_value``.
        fill_value: Timestamp value used when ``strategy='constant'``.

    Returns:
        Series with missing values filled.

    Raises:
        ValueError: If ``strategy`` is not one of the accepted values.
    """
    if not col_series.isnull().any():
        return col_series.copy()

    if strategy == "ffill":
        result = col_series.ffill().bfill()
    elif strategy == "bfill":
        result = col_series.bfill().ffill()
    elif strategy == "median":
        if not col_series.isnull().all():
            median_val = col_series.dropna().astype("int64").median()
            result = col_series.fillna(
                col_series.dtype.type(int(median_val))
                if hasattr(col_series.dtype, "type")
                else median_val
            )
        else:
            result = col_series.fillna(fill_value) if fill_value is not None else col_series.copy()
    elif strategy == "constant":
        result = col_series.fillna(fill_value) if fill_value is not None else col_series.copy()
    else:
        raise ValueError(
            f"Unknown strategy '{strategy}'. "
            "Choose from 'ffill', 'bfill', 'median', 'constant'."
        )

    return result


def fillna_by_coltype(
    col_series: Series,
    col_type: ColType,
    **kwargs,
) -> Series:
    """
    Fill missing values based on column type.
    Dispatches to the appropriate fill function according to ``col_type`` and
    forwards any extra keyword arguments to it.

    Args:
        col_series: pandas Series to fill.
        col_type: Semantic type of the column (``ColType`` enum).
        **kwargs: Extra keyword arguments forwarded to the underlying fill
            function:

            - ``NUMERICAL``: ``strategy``, ``fill_value`` — see
              :func:`fillna_numerical`.
            - ``CATEGORICAL``: ``fill_value`` — see
              :func:`fillna_categorical`.
            - ``TEXT``: ``fill_value`` — see :func:`fillna_text`.
            - ``TIMESTAMP``: ``strategy``, ``fill_value`` — see
              :func:`fillna_timestamp`.
            - ``BINARY``: no extra arguments.

    Returns:
        Series with missing values filled. Returns a copy unchanged for
        unrecognised column types.
    """
    if col_type == ColType.NUMERICAL:
        return fillna_numerical(col_series, **kwargs)
    elif col_type == ColType.CATEGORICAL:
        return fillna_categorical(col_series, **kwargs)
    elif col_type == ColType.BINARY:
        return fillna_binary(col_series)
    elif col_type == ColType.TEXT:
        return fillna_text(col_series, **kwargs)
    elif col_type == ColType.TIMESTAMP:
        return fillna_timestamp(col_series, **kwargs)
    else:
        return col_series.copy()
