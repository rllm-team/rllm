from typing import List, Mapping, Optional, Sequence, Tuple

import pandas
from pandas import Series
from torch import Tensor

from rllm.types import ColType

def encode_categorical(
    col_series: Series,
    missing_values: Optional[Sequence] = None,
) -> Tuple[Series, dict]:
    """Encode a categorical column to consecutive integer codes.

    Uses ``pandas.factorize`` internally, which is faster than
    ``LabelEncoder`` and avoids an extra sklearn dependency for this step.
    Missing values are mapped to ``-1`` in the output.

    Args:
        col_series: Series with categorical data.
        missing_values: Extra values to treat as missing in addition to
            ``NaN``.  Defaults to ``[-1, "-1"]``.

    Returns:
        Tuple of (encoded ``Series[int64]``, mapping dict ``{code: label}``).
    """
    _default_missing_values = [-1, "-1"]
    if missing_values is None:
        missing_values = _default_missing_values

    col_copy = col_series.copy()

    # Build missing mask: NaN + user-specified sentinels.
    missing_mask = col_copy.isna()
    for sentinel in missing_values:
        try:
            missing_mask = missing_mask | col_copy.eq(sentinel)
        except TypeError:
            pass
    # Also catch string representations after stripping whitespace.
    str_vals = col_copy.astype("string").str.strip()
    for sentinel in missing_values:
        missing_mask = missing_mask | str_vals.eq(str(sentinel).strip())

    # Cast to str (matching original LabelEncoder behaviour for mixed-type columns),
    # then factorize only non-missing values so lengths match the masked assignment.
    valid_values = col_copy.loc[~missing_mask].astype(str)
    codes, uniques = pandas.factorize(valid_values, sort=True)
    encoded = pandas.Series(-1, index=col_copy.index, dtype="int64", name=col_copy.name)
    encoded.loc[~missing_mask] = codes.astype("int64")

    label_map: dict = dict(enumerate(uniques))
    return encoded, label_map


def convert_binary(
    col_series: Series,
    true_values: Optional[Sequence[str]] = None,
) -> Series:
    """Convert a binary column to 0/1 integer format.

    Args:
        col_series: Series with binary data.
        true_values: Strings (case-insensitive) that map to 1.
            Defaults to ``["1", "yes", "true", "t", "y"]``.

    Returns:
        Series with values 0 or 1 (``int64``).
    """
    _default_true_values = frozenset(["1", "yes", "true", "t", "y"])
    true_set = (
        frozenset(v.lower() for v in true_values)
        if true_values is not None
        else _default_true_values
    )
    return col_series.astype(str).str.lower().isin(true_set).astype("int64")


def dict_to_df(
    data_dict: Mapping[ColType, Tensor],
    categorical_columns: Optional[List[str]] = None,
    numerical_columns: Optional[List[str]] = None,
    binary_columns: Optional[List[str]] = None,
) -> pandas.DataFrame:
    """Reconstruct a DataFrame from a feature-tensor dict.

    Args:
        data_dict: Mapping from :class:`ColType` to a CPU or GPU tensor.
        categorical_columns: Column names for ``ColType.CATEGORICAL``.
        numerical_columns: Column names for ``ColType.NUMERICAL``.
        binary_columns: Column names for ``ColType.BINARY``.

    Returns:
        Concatenated DataFrame.  Raises ``ValueError`` if no recognised
        column type is present.
    """
    col_spec = [
        (ColType.CATEGORICAL, categorical_columns),
        (ColType.NUMERICAL,   numerical_columns),
        (ColType.BINARY,      binary_columns),
    ]

    parts = [
        pandas.DataFrame(tensor.cpu().numpy(), columns=cols)
        for col_type, cols in col_spec
        if (tensor := data_dict.get(col_type)) is not None
    ]

    if not parts:
        raise ValueError(
            "data_dict contains no recognised column types "
            f"({[ct for ct, _ in col_spec]})."
        )

    return pandas.concat(parts, axis=1)
