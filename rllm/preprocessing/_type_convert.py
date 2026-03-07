from typing import Optional, Dict

import pandas
from pandas import Series
from sklearn.preprocessing import LabelEncoder

from rllm.types import ColType


def encode_categorical(col_series: Series) -> tuple[Series, LabelEncoder]:
    """
    Encode categorical column using LabelEncoder.
    Missing values are preserved as -1 in output.

    Args:
        col_series: pandas Series with categorical data.
            NaN, -1 and "-1" are treated as missing sentinels.

    Returns:
        Tuple of (encoded Series[int64], LabelEncoder instance).
    """
    col_copy = col_series.copy()
    # Treat NaN, numeric -1, and string "-1" (with optional surrounding spaces)
    # as missing sentinels.
    col_as_str = col_copy.astype("string").str.strip()
    missing_mask = col_copy.isna() | col_copy.eq(-1) | col_as_str.eq("-1")
    # Encode non-missing values (cast to str to avoid mixed dtype issues).
    col_fit = col_copy[~missing_mask].astype(str)
    encoder = LabelEncoder()
    encoded = pandas.Series(-1, index=col_copy.index, dtype="int64", name=col_copy.name)
    if len(col_fit) > 0:
        labels = encoder.fit_transform(col_fit).astype("int64")
        encoded.loc[~missing_mask] = labels
    return encoded, encoder


def convert_binary(
    col_series: Series, binary_map: Optional[list[str]] = None
) -> Series:
    """
    Convert binary column to 0/1 format.

    Args:
        col_series: pandas Series with binary data
        binary_map: List of strings that indicate True/1 values
                   (default: ["1", "yes", "true", "t", "y"])

    Returns:
        Series with binary values (0 or 1)
    """
    if binary_map is None:
        binary_map = ["1", "yes", "true", "t", "y"]

    col_copy = col_series.astype(str).map(lambda x: 1 if x.lower() in binary_map else 0)
    return col_copy


def dict_to_df(
    data_dict: Dict[str, list], categorical_columns, numerical_columns, binary_columns
) -> pandas.DataFrame:
    parts = []
    if data_dict.get(ColType.CATEGORICAL) is not None:
        parts.append(
            pandas.DataFrame(
                data_dict[ColType.CATEGORICAL].cpu().numpy(),
                columns=categorical_columns,
            )
        )
    if data_dict.get(ColType.NUMERICAL) is not None:
        parts.append(
            pandas.DataFrame(
                data_dict[ColType.NUMERICAL].cpu().numpy(),
                columns=numerical_columns,
            )
        )
    if data_dict.get(ColType.BINARY) is not None:
        parts.append(
            pandas.DataFrame(
                data_dict[ColType.BINARY].cpu().numpy(),
                columns=binary_columns,
            )
        )

    df = pandas.concat(parts, axis=1)
    return df
