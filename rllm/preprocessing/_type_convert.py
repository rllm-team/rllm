from typing import Optional, Dict
from pandas import Series
from sklearn.preprocessing import LabelEncoder

from rllm.types import ColType

# Default binary mapping: values that represent True/1
DEFAULT_BINARY_MAP = ["1", "yes", "true", "t", "y"]


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
        binary_map = DEFAULT_BINARY_MAP

    col_copy = col_series.astype(str).map(lambda x: 1 if x.lower() in binary_map else 0)
    return col_copy


def convert_categorical_to_text(
    col_types: Dict[str, ColType], target_col: Optional[str] = None
) -> Dict[str, ColType]:
    """Convert CATEGORICAL columns (except target_col) to TEXT type.

    This is useful for models like TransTab that require text tokenization
    for categorical features.

    Args:
        col_types: Original column type mapping
        target_col: Target column name (should remain CATEGORICAL)

    Returns:
        Modified column type mapping with CATEGORICAL -> TEXT conversion
    """
    converted_types = {}
    for col_name, col_type in col_types.items():
        if col_type == ColType.CATEGORICAL and col_name != target_col:
            converted_types[col_name] = ColType.TEXT
        else:
            converted_types[col_name] = col_type
    return converted_types
