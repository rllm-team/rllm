from typing import Optional
from pandas import Series

import torch

from rllm.preprocessing._fillna import fillna_by_coltype
from rllm.preprocessing._type_convert import (
    encode_categorical,
    convert_binary,
)
from rllm.preprocessing._text_tokenize import (
    TokenizerConfig,
    process_tokenized_column,
    tokenize_merged_cols,
)
from rllm.preprocessing._word_embedding import (
    TextEmbedderConfig,
    embed_text_column,
)
from rllm.types import ColType


def df_to_tensor(
    df,
    col_types,
    target_col=None,
    text_embedder_config: Optional[TextEmbedderConfig] = None,
    tokenizer_config: Optional[TokenizerConfig] = None,
    concat: bool = True,
    cat_hardcode: bool = True,
):
    r"""Convert DataFrame to tensor dictionary.

    Args:
        df: Input DataFrame
        col_types: Dictionary mapping column names to column types
        target_col: Name of target column
        text_embedder_config: Configuration for text embedding
        tokenizer_config: Configuration for tokenization

    Returns:
        tuple: (feat_dict, y) where feat_dict contains feature tensors by column type,
            and y is the target tensor (None if no target_col)
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
):
    col_copy = col.copy()
    if col_type == ColType.NUMERICAL:
        col_copy = fillna_by_coltype(col_copy, ColType.NUMERICAL)
        return torch.tensor(col_copy.values.astype(float), dtype=torch.float32).reshape(
            -1, 1
        )

    elif col_type == ColType.CATEGORICAL:
        col_copy = fillna_by_coltype(col_copy, ColType.CATEGORICAL)
        col_copy, _ = encode_categorical(col_copy)
        return torch.tensor(col_copy.values.astype(float), dtype=torch.float32).reshape(
            -1, 1
        )

    elif col_type == ColType.BINARY:
        col_copy = fillna_by_coltype(col_copy, ColType.BINARY)
        col_copy = convert_binary(col_copy)
        return torch.tensor(col_copy.values.astype(float), dtype=torch.float32).reshape(
            -1, 1
        )

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
