import pandas as pd

from rllm.types import ColType


def read_csv_with_fallback_encodings(
    path: str,
    encodings: tuple[str, ...] = (
        "utf-8",
        "utf-8-sig",
        "gbk",
        "gb18030",
        "latin1",
    ),
    **kwargs,
) -> pd.DataFrame:
    r"""Read a CSV file by trying multiple encodings sequentially.

    Args:
        path (str): Path to the CSV file.
        encodings (tuple[str, ...]): Candidate encodings to try in order.
        **kwargs: Additional keyword arguments forwarded to :func:`pandas.read_csv`.

    Returns:
        pandas.DataFrame: Loaded DataFrame.

    Raises:
        ValueError: If the file cannot be loaded with any of the provided encodings.
    """
    last_err: Exception | None = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except Exception as e:  # pragma: no cover - best-effort fallback
            last_err = e
            continue
    raise ValueError(f"Failed to load CSV with any supported encoding: {last_err}")


def clean_df_by_col_types(df: pd.DataFrame, col_types: dict) -> pd.DataFrame:
    r"""Lightweight :class:`pandas.DataFrame` cleaning based on column types.

    This helper keeps dataset-specific logic minimal and defers most
    handling to the preprocessing utilities (e.g., :func:`df_to_tensor`).

    Current behavior:

    - Strip whitespace from column names.
    - For :obj:`CATEGORICAL` columns: normalize common string missing tokens
      (e.g., ``"nan"``, ``"None"``) to :obj:`pandas.NA`.
    - For :obj:`NUMERICAL` columns: keep values as-is.

    Args:
        df (pandas.DataFrame): The input DataFrame to clean.
        col_types (dict): A mapping from column name to :obj:`ColType`.

    Returns:
        pandas.DataFrame: A cleaned copy of the input DataFrame.
    """
    df = df.copy()
    df.columns = df.columns.str.strip()

    missing_tokens = {"nan", "None", "NaN", "<NA>"}

    for col_name, col_type in col_types.items():
        if col_name not in df.columns:
            continue

        if col_type == ColType.CATEGORICAL:
            df[col_name] = df[col_name].replace(list(missing_tokens), pd.NA)

    return df

