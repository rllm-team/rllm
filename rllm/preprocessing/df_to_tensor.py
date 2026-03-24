from typing import Optional, Sequence, Union, Tuple
from pandas import Series

import torch
from torch import Tensor

from rllm.preprocessing._fillna import fillna_by_coltype
from rllm.preprocessing._type_convert import (
    encode_categorical,
    convert_binary,
)
from rllm.preprocessing.data_clean import to_numeric_by_column
from rllm.preprocessing.text_tokenize import (
    TokenizerConfig,
    process_tokenized_column,
    tokenize_merged_cols,
)
from rllm.preprocessing.word_embedding import (
    TextEmbedderConfig,
    embed_text_column,
)
from rllm.preprocessing.timestamp import TimestampPreprocessor
from rllm.types import ColType

# df_to_tensor应该是一个可配置的类/函数
def df_to_tensor(
    df,
    col_types,
    target_col=None,
    text_embedder_config: Optional[TextEmbedderConfig] = None,
    tokenizer_config: Optional[TokenizerConfig] = None,
    timestamp_format: Optional[str] = None,
    timestamp_fields: Optional[Sequence[str]] = None,
    categorical_missing_values: Optional[Sequence] = None,
    binary_true_values: Optional[Sequence[str]] = None,
    concat: bool = True,
    cat_hardcode: bool = True,
):
    r"""Convert DataFrame to tensor dictionary.

    Args:
        df: Input DataFrame
        col_types: Dictionary mapping column names to column types
        target_col: Name of target column
        text_embedder_config: Configuration for text embedding
        tokenizer_config: Configuration for tokenization; if provided and
            ``tokenize_combine`` is True, all TEXT columns are jointly tokenized
            and stored as a single entry in ``feat_dict[ColType.TEXT]``.
        timestamp_format: Optional format string for parsing ``TIMESTAMP``
            columns. ``None`` lets ``pd.to_datetime`` infer the format.
        timestamp_fields: Optional list of time components to extract from
            ``TIMESTAMP`` columns (subset of ``["YEAR", "MONTH", "DAY",
            "DAYOFWEEK", "HOUR", "MINUTE", "SECOND"]``).
        categorical_missing_values: Optional extra values treated as missing
            when encoding categorical columns. Passed to
            :func:`encode_categorical`.
        binary_true_values: Optional list of string values that should be
            interpreted as 1 in binary columns. Passed to
            :func:`convert_binary`.
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

    # 1. Iterate each column
    feat_dict = {}
    y = None

    merged_token = None
    if tokenizer_config is not None and tokenizer_config.tokenize_combine:
        merged_token = tokenize_merged_cols(
            df=df,
            col_types=col_types,
            tokenizer_config=tokenizer_config,
            target_col=target_col,
        )  # (ids [N,L], mask [N,L]) or None

    for col_name, col_type in col_types.items():
        if merged_token is not None and col_type == ColType.TEXT:
            continue
        # 2. Get column tensor shape: (n, 1) for cat/num, (n, d) for text
        col_tensor = _generate_column_tensor(
            col=df[col_name],
            col_type=col_type,
            col_name=col_name,
            text_embedder_config=text_embedder_config,
            tokenizer_config=tokenizer_config,
            timestamp_format=timestamp_format,
            timestamp_fields=timestamp_fields,
            categorical_missing_values=categorical_missing_values,
            binary_true_values=binary_true_values,
        )
        # 3. Update feat dict
        if col_name == target_col:
            # Only need one-dimensional
            y = col_tensor.squeeze()
            continue
        if col_type not in feat_dict.keys():
            feat_dict[col_type] = []
        feat_dict[col_type].append(col_tensor)

    # 4. Concat column tensors
    if concat:
        for col_type, xs in feat_dict.items():
            if col_type == ColType.TEXT:
                # Check if TEXT columns are tokenized or embedded
                if tokenizer_config is not None and isinstance(xs[0], tuple):
                    # xs: List[(ids [N,L], mask [N,L])]
                    ids = torch.stack([t[0] for t in xs], dim=1).long()  # [N, C_tok, L]
                    mask = torch.stack(
                        [t[1] for t in xs], dim=1
                    ).long()  # [N, C_tok, L]
                    feat_dict[col_type] = (ids, mask)
                else:
                    # Embedded text: stack along dim=1
                    feat_dict[col_type] = torch.stack(xs, dim=1)
            elif col_type == ColType.TIMESTAMP:
                # As diff timestamp features represent different aspects,
                # we keep them separate along dim=1
                feat_dict[col_type] = torch.stack(xs, dim=1)  # [N, 1, 7]
            else:
                feat_dict[col_type] = torch.cat(xs, dim=-1)


    if merged_token is not None:
        feat_dict[ColType.TEXT] = merged_token  # (ids [N,L], mask [N,L])

    # 5. Change hard-coding here
    if cat_hardcode and ColType.CATEGORICAL in feat_dict.keys():
        feat_dict[ColType.CATEGORICAL] = feat_dict[ColType.CATEGORICAL].int()
    return feat_dict, y


def _generate_column_tensor(
    col: Series,
    col_type: ColType,
    col_name: str,
    text_embedder_config: Optional[TextEmbedderConfig] = None,
    tokenizer_config: Optional[TokenizerConfig] = None,
    timestamp_format: Optional[str] = None,
    timestamp_fields: Optional[Sequence[str]] = None,
    categorical_missing_values: Optional[Sequence] = None,
    binary_true_values: Optional[Sequence[str]] = None,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    col_copy = col.copy()
    if col_type == ColType.NUMERICAL:
        col_copy = to_numeric_by_column(col_copy)
        col_copy = fillna_by_coltype(col_copy, ColType.NUMERICAL)
        return torch.tensor(col_copy.values.astype(float), dtype=torch.float32).reshape(
            -1, 1
        )

    elif col_type == ColType.CATEGORICAL:
        col_copy = fillna_by_coltype(col_copy, ColType.CATEGORICAL)
        col_copy, _ = encode_categorical(
            col_copy,
            missing_values=categorical_missing_values,
        )
        return torch.tensor(col_copy.values.astype(float), dtype=torch.float32).reshape(
            -1, 1
        )

    elif col_type == ColType.BINARY:
        col_copy = fillna_by_coltype(col_copy, ColType.BINARY)
        col_copy = convert_binary(
            col_copy,
            true_values=binary_true_values,
        )
        return torch.tensor(col_copy.values.astype(float), dtype=torch.float32).reshape(
            -1, 1
        )

    # TODO: (Feiyu Pan) If table contains two text columns, which require different
    # processing (one embedding, one tokenization), current design cannot handle it.
    elif col_type == ColType.TEXT:
        # Determine processing mode based on config
        if tokenizer_config is not None:
            # Tokenize mode
            assert (
                tokenizer_config.tokenizer is not None
            ), "Need a tokenizer_config with a valid tokenizer for TEXT column!"

            input_ids, attention_mask = process_tokenized_column(
                col_series=col_copy,
                col_name=col_name,
                tokenizer_config=tokenizer_config,
                include_colname=tokenizer_config.include_colname,
                name_value_sep=tokenizer_config.name_value_sep,
            )
            return (input_ids, attention_mask)
        else:
            # Embedding mode
            return embed_text_column(col_copy, text_embedder_config)

    elif col_type == ColType.TIMESTAMP:
        # Default: [Batch, 7], (year, month, day, dayofweek, hour, minute, second).
        # Pass timestamp_format / timestamp_fields to control parsing and output dims,
        # e.g. timestamp_fields=["YEAR","MONTH","DAY"] → [Batch, 3].
        preprocessor = TimestampPreprocessor(
            format=timestamp_format,
            fields=timestamp_fields,
        )
        return preprocessor(col_copy)

    else:
        raise NotImplementedError(
            f"Column type {col_type} not implemented in _generate_column_tensor."
        )