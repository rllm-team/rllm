from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple, Union

from pandas import DataFrame, Series
import torch
from torch import Tensor

from rllm.preprocessing.fillna import FillNAConfig, fillna_by_coltype
from rllm.preprocessing._type_convert import encode_categorical, convert_binary
from rllm.preprocessing.data_clean import to_numeric_by_column
from rllm.preprocessing.text_tokenize import (
    TokenizerConfig,
    process_tokenized_column,
    tokenize_merged_cols,
)
from rllm.preprocessing.word_embedding import TextEmbedderConfig, embed_text_column
from rllm.preprocessing.timestamp import TimestampPreprocessor
from rllm.types import ColType

TextToken = Tuple[Tensor, Tensor]
ColumnTensor = Union[Tensor, TextToken]
FeatureDict = Dict[ColType, Union[Tensor, TextToken]]


def _to_float_col_tensor(col_series: Series) -> Tensor:
    """Convert Series to float32 tensor with shape [N, 1]."""
    return torch.from_numpy(col_series.to_numpy(dtype="float32")).view(-1, 1)


def _build_scalar_tensor(
    col: Series,
    col_type: ColType,
    *,
    fillna_config: FillNAConfig,
    categorical_missing_values: Optional[Sequence] = None,
    binary_true_values: Optional[Sequence[str]] = None,
) -> Tensor:
    """Build tensor for NUMERICAL/CATEGORICAL/BINARY column types."""
    if col_type == ColType.NUMERICAL:
        cleaned = to_numeric_by_column(col)
        filled = fillna_by_coltype(
            cleaned,
            ColType.NUMERICAL,
            strategy=fillna_config.numerical_strategy,
            fill_value=fillna_config.numerical_fill_value,
        )
        return _to_float_col_tensor(filled)

    if col_type == ColType.CATEGORICAL:
        filled = fillna_by_coltype(
            col,
            ColType.CATEGORICAL,
            fill_value=fillna_config.categorical_fill_value,
        )
        encoded, _ = encode_categorical(
            filled,
            missing_values=categorical_missing_values,
        )
        return _to_float_col_tensor(encoded)

    if col_type == ColType.BINARY:
        filled = fillna_by_coltype(col, ColType.BINARY)
        converted = convert_binary(filled, true_values=binary_true_values)
        return _to_float_col_tensor(converted)

    raise ValueError(f"Unsupported type in _build_scalar_tensor: {col_type}")


def _build_text_tensor(
    col: Series,
    col_name: str,
    *,
    fillna_config: FillNAConfig,
    tokenizer_config: Optional[TokenizerConfig] = None,
    text_embedder_config: Optional[TextEmbedderConfig] = None,
) -> ColumnTensor:
    """Build tensor for TEXT column.

    Returns:
        Tokenized mode: ``(input_ids [N, L], attention_mask [N, L])``
        Embedded mode:  ``[N, D]``
    """
    filled = fillna_by_coltype(
        col,
        ColType.TEXT,
        fill_value=fillna_config.text_fill_value,
    )

    if tokenizer_config is not None:
        assert (
            tokenizer_config.tokenizer is not None
        ), "Need a tokenizer_config with a valid tokenizer for TEXT column!"
        return process_tokenized_column(
            col_series=filled,
            col_name=col_name,
            tokenizer_config=tokenizer_config,
            include_colname=tokenizer_config.include_colname,
            name_value_sep=tokenizer_config.name_value_sep,
        )

    return embed_text_column(filled, text_embedder_config)


def _build_timestamp_tensor(
    col: Series,
    *,
    fillna_config: FillNAConfig,
    timestamp_format: Optional[str] = None,
    timestamp_fields: Optional[Sequence[str]] = None,
) -> Tensor:
    """Build tensor for TIMESTAMP column.

    Returns:
        ``[N, F]`` where F is the number of extracted time fields
        (default F=7: year, month, day, dayofweek, hour, minute, second).
    """
    filled = fillna_by_coltype(
        col,
        ColType.TIMESTAMP,
        strategy=fillna_config.timestamp_strategy,
        fill_value=fillna_config.timestamp_fill_value,
    )
    preprocessor = TimestampPreprocessor(
        format=timestamp_format,
        fields=timestamp_fields,
    )
    return preprocessor(filled)


def _generate_column_tensor(
    col: Series,
    col_type: ColType,
    col_name: str,
    *,
    fillna_config: FillNAConfig,
    categorical_missing_values: Optional[Sequence] = None,
    binary_true_values: Optional[Sequence[str]] = None,
    tokenizer_config: Optional[TokenizerConfig] = None,
    text_embedder_config: Optional[TextEmbedderConfig] = None,
    timestamp_format: Optional[str] = None,
    timestamp_fields: Optional[Sequence[str]] = None,
) -> ColumnTensor:
    if col_type in (ColType.NUMERICAL, ColType.CATEGORICAL, ColType.BINARY):
        return _build_scalar_tensor(
            col,
            col_type,
            fillna_config=fillna_config,
            categorical_missing_values=categorical_missing_values,
            binary_true_values=binary_true_values,
        )
    if col_type == ColType.TEXT:
        return _build_text_tensor(
            col,
            col_name,
            fillna_config=fillna_config,
            tokenizer_config=tokenizer_config,
            text_embedder_config=text_embedder_config,
        )
    if col_type == ColType.TIMESTAMP:
        return _build_timestamp_tensor(
            col,
            fillna_config=fillna_config,
            timestamp_format=timestamp_format,
            timestamp_fields=timestamp_fields,
        )
    raise NotImplementedError(
        f"Column type {col_type} not implemented in _generate_column_tensor."
    )


def _aggregate_type_tensors(
    type_tensors: Dict[ColType, List[ColumnTensor]],
    *,
    concat: bool,
) -> FeatureDict:
    if not concat:
        return dict(type_tensors)  # type: ignore[return-value]

    out: FeatureDict = {}
    for col_type, xs in type_tensors.items():
        if not xs:
            continue

        if col_type == ColType.TEXT:
            # tokenized: List[(ids [N, L], mask [N, L])] -> ([N, C, L], [N, C, L])
            if isinstance(xs[0], tuple):
                tokens = xs  # type: ignore[assignment]
                ids = torch.stack([t[0] for t in tokens], dim=1).long()
                mask = torch.stack([t[1] for t in tokens], dim=1).long()
                out[col_type] = (ids, mask)
            else:
                # embedded text: List[[N, D]] -> [N, C, D]
                out[col_type] = torch.stack(xs, dim=1)  # type: ignore[arg-type]
        elif col_type == ColType.TIMESTAMP:
            # As diff timestamp features represent different aspects,
            # we keep them separate along dim=1: [N, C, F]
            out[col_type] = torch.stack(xs, dim=1)  # type: ignore[arg-type]
        else:
            # NUMERICAL/CATEGORICAL/BINARY: List[[N,1]] -> [N, C]
            out[col_type] = torch.cat(xs, dim=-1)  # type: ignore[arg-type]
    return out


def df_to_tensor(
    df: DataFrame,
    col_types: Dict[str, ColType],
    target_col: Optional[str] = None,
    fillna_config: Optional[FillNAConfig] = None,
    categorical_missing_values: Optional[Sequence] = None,
    binary_true_values: Optional[Sequence[str]] = None,
    tokenizer_config: Optional[TokenizerConfig] = None,
    text_embedder_config: Optional[TextEmbedderConfig] = None,
    timestamp_format: Optional[str] = None,
    timestamp_fields: Optional[Sequence[str]] = None,
    concat: bool = True,
    cat_hardcode: bool = True,
) -> Tuple[FeatureDict, Optional[Tensor]]:
    r"""Convert DataFrame to tensor dictionary.

    Args:
        df: Input DataFrame
        col_types: Dictionary mapping column names to column types
        target_col: Name of target column
        fillna_config: Fill-NA configuration shared by all supported column
            types. When ``None``, :class:`FillNAConfig` defaults are used.
        categorical_missing_values: Optional extra values treated as missing
            when encoding categorical columns. Passed to
            :func:`encode_categorical`.
        binary_true_values: Optional list of string values that should be
            interpreted as 1 in binary columns. Passed to
            :func:`convert_binary`.
        tokenizer_config: Configuration for tokenization; if provided and
            ``tokenize_combine`` is True, all TEXT columns are jointly tokenized
            and stored as a single entry in ``feat_dict[ColType.TEXT]``.
        text_embedder_config: Configuration for text embedding.
        timestamp_format: Optional format string for parsing ``TIMESTAMP``
            columns. ``None`` lets ``pd.to_datetime`` infer the format.
        timestamp_fields: Optional list of time components to extract from
            ``TIMESTAMP`` columns (subset of ``["YEAR", "MONTH", "DAY",
            "DAYOFWEEK", "HOUR", "MINUTE", "SECOND"]``).
        concat: Whether to concatenate/stack features of the same column type
            (e.g., numerical and categorical along the last dim, text and
            timestamp along the feature/channel dim).
        cat_hardcode: Whether to cast categorical features to integer type.

    Returns:
        tuple: (feat_dict, y) where feat_dict contains feature tensors by column type,
            and y is the target tensor (None if no target_col). When TEXT columns
            are tokenized, the corresponding value is a tuple of
            ``(input_ids, attention_mask)``; otherwise it is an embedded tensor.
    """
    cfg = fillna_config or FillNAConfig()
    y: Optional[Tensor] = None
    merged_token: Optional[TextToken] = None

    if tokenizer_config is not None and tokenizer_config.tokenize_combine:
        merged_token = tokenize_merged_cols(
            df=df,
            col_types=col_types,
            tokenizer_config=tokenizer_config,
            target_col=target_col,
        )

    grouped: Dict[ColType, List[ColumnTensor]] = defaultdict(list)
    for col_name, col_type in col_types.items():
        if merged_token is not None and col_type == ColType.TEXT:
            continue

        col_tensor = _generate_column_tensor(
            col=df[col_name],
            col_type=col_type,
            col_name=col_name,
            fillna_config=cfg,
            categorical_missing_values=categorical_missing_values,
            binary_true_values=binary_true_values,
            tokenizer_config=tokenizer_config,
            text_embedder_config=text_embedder_config,
            timestamp_format=timestamp_format,
            timestamp_fields=timestamp_fields,
        )

        if col_name == target_col:
            if isinstance(col_tensor, tuple):
                raise ValueError(
                    "Target column cannot be tokenized TEXT tuple; "
                    "please set target_col to numerical/categorical/binary-like output."
                )
            y = col_tensor.squeeze()
            continue

        grouped[col_type].append(col_tensor)

    feat_dict = _aggregate_type_tensors(grouped, concat=concat)

    if merged_token is not None:
        feat_dict[ColType.TEXT] = merged_token

    if cat_hardcode and ColType.CATEGORICAL in feat_dict:
        cat_val = feat_dict[ColType.CATEGORICAL]
        if isinstance(cat_val, tuple):
            raise ValueError("Categorical feature should not be token tuple.")
        if isinstance(cat_val, list):
            feat_dict[ColType.CATEGORICAL] = [t.int() for t in cat_val]
        else:
            feat_dict[ColType.CATEGORICAL] = cat_val.int()

    return feat_dict, y
