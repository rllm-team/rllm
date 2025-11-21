from typing import Optional, Dict

import pandas
from pandas import Series
from sklearn.preprocessing import LabelEncoder

from rllm.types import ColType


def encode_categorical(col_series: Series) -> tuple[Series, LabelEncoder]:
    """
    Encode categorical column using LabelEncoder.
    Missing values should be marked as -1 and will be preserved.

    Args:
        col_series: pandas Series with categorical data (missing values as -1)

    Returns:
        Tuple of (encoded Series, LabelEncoder instance)
    """
    col_copy = col_series.copy()
    # Encode non-missing values
    col_fit = col_copy[col_copy != -1]
    encoder = LabelEncoder()
    labels = encoder.fit_transform(col_fit)
    col_copy[col_copy != -1] = labels
    return col_copy, encoder


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
