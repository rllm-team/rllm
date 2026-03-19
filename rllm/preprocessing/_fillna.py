from pandas import Series

from rllm.types import ColType


def fillna_numerical(col_series: Series) -> Series:
    """
    Fill missing values for numerical columns with mean value.
    If all values are NaN, fill with 0.

    Args:
        col_series: pandas Series with numerical data

    Returns:
        Series with filled values
    """
    col_copy = col_series.copy()
    if col_copy.isnull().any():
        # Use mean for fillna, or 0 if all values are NaN
        mean_value = col_copy.mean() if not col_copy.isnull().all() else 0.0
        col_copy.fillna(mean_value, inplace=True)
    return col_copy


def fillna_categorical(col_series: Series) -> Series:
    """
    Fill missing values for categorical columns with -1.

    Args:
        col_series: pandas Series with categorical data

    Returns:
        Series with filled values
    """
    col_copy = col_series.copy()
    if col_copy.isnull().any():
        col_copy.fillna(-1, inplace=True)
    return col_copy


def fillna_binary(col_series: Series) -> Series:
    """
    Fill missing values for binary columns with mode.

    Args:
        col_series: pandas Series with binary data

    Returns:
        Series with filled values
    """
    col_copy = col_series.copy()
    if col_copy.isnull().any():
        col_copy.fillna(col_copy.mode()[0], inplace=True)
    return col_copy


def fillna_by_coltype(
    col_series: Series,
    col_type: ColType,
) -> Series:
    """
    Fill missing values based on column type.

    Args:
        col_series: pandas Series to fill
        col_type: Type of the column

    Returns:
        Series with filled values
    """
    if col_type == ColType.NUMERICAL:
        return fillna_numerical(col_series)
    elif col_type == ColType.CATEGORICAL:
        return fillna_categorical(col_series)
    elif col_type == ColType.BINARY:
        return fillna_binary(col_series)
    else:
        return col_series.copy()
