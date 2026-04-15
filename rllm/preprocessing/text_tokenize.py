from __future__ import annotations
from typing import Any, Callable, Optional
from dataclasses import dataclass
from collections.abc import Mapping

import numpy as np
import pandas
from pandas import Series, DataFrame
import torch

from rllm.types import ColType


@dataclass
class TokenizerConfig:
    """Configuration for text tokenization.

    Args:
        tokenizer (Callable[[list[str]], Any]): Tokenizer callable that accepts
            a list of strings.
        batch_size (Optional[int]): Optional mini-batch size for tokenization.
        pad_token_id (int): Padding token ID used when masks are generated.
        tokenize_combine (bool): Whether to tokenize all text columns as one
            merged string per row.
        include_colname (bool): Whether to prepend column names to cell values.
        save_colname_token_ids (bool): Whether to cache tokenized column-name
            ids for downstream reuse.
        segment_sep (str): Separator between merged text segments.
        name_value_sep (str): Separator between column name and text value.
    """

    tokenizer: Callable[[list[str]], Any]
    batch_size: Optional[int] = None
    pad_token_id: int = 0
    tokenize_combine: bool = True
    include_colname: bool = True
    save_colname_token_ids: bool = False
    segment_sep: str = " "
    name_value_sep: str = " "


def process_tokenized_column(
    col_series: Series,
    col_name: str,
    tokenizer_config: "TokenizerConfig",
    include_colname: bool = True,
    name_value_sep: str = " ",
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Tokenize a single text column.

    Args:
        col_series (Series): Input text column.
        col_name (str): Column name.
        tokenizer_config (TokenizerConfig): Tokenizer configuration.
        include_colname (bool): Whether to prepend the column name.
        name_value_sep (str): Separator between column name and value.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: ``(input_ids, attention_mask)``,
        both with shape :math:`(N, L)`.
    """
    col_str = col_series.astype(str).fillna("")

    if include_colname:
        col_list = [f"{col_name}{name_value_sep}{v}" for v in col_str.tolist()]
    else:
        col_list = col_str.tolist()

    input_ids, attention_mask = tokenize_strings(
        col_list,
        tokenizer_config.tokenizer,
        tokenizer_config.pad_token_id,
        standardize_tokenizer_output,
        tokenizer_config.batch_size,
    )
    return input_ids.long(), attention_mask.long()


def tokenize_strings(
    seqs: list[str],
    tokenizer: Callable,
    pad_token_id: int,
    standardize_func: Callable,
    batch_size: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Tokenize a list of strings.

    Args:
        seqs (list[str]): Strings to tokenize.
        tokenizer (Callable): Tokenizer callable.
        pad_token_id (int): Padding token ID.
        standardize_func (Callable): Function that normalizes tokenizer output
            into ids and masks.
        batch_size (Optional[int]): Mini-batch size. ``None`` means one shot.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: ``(input_ids, attention_mask)``,
        both with shape :math:`(B, L)`.
    """
    if batch_size is None:
        input_ids, attention_mask = standardize_func(tokenizer(seqs), pad_token_id)
        return input_ids.long(), attention_mask.long()

    ids_list, mask_list = [], []
    for i in range(0, len(seqs), batch_size):
        _ids, _mask = standardize_func(
            tokenizer(seqs[i : i + batch_size]), pad_token_id
        )
        ids_list.append(_ids)
        mask_list.append(_mask)
    return torch.cat(ids_list, dim=0).long(), torch.cat(mask_list, dim=0).long()


def standardize_tokenizer_output(
    tok_output, pad_token_id: int
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Standardize tokenizer outputs into ``(input_ids, attention_mask)``.

    Supported input formats:
    - Mapping (e.g., transformers.BatchEncoding): keys "input_ids", optional "attention_mask"
    - Tuple/List: (input_ids, attention_mask) or List[Encoding] or List[List[int]]
    - Single Encoding/EncodingFast: .ids, optional .attention_mask
    - Raw ids only: List[int] | List[List[int]] | np.ndarray | torch.Tensor

    Behavior:
    - Converts to 2D tensors [B, L]; ragged sequences are padded with `pad_token_id`.
    - If attention_mask is missing, it is derived as (input_ids != pad_token_id).
    - Ensures input_ids and attention_mask share the same shape and dtype=torch.long.

    Args:
        tok_output: Raw output from a tokenizer.
        pad_token_id (int): Padding token ID.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: ``(input_ids, attention_mask)``,
        both with shape :math:`(B, L)` and dtype ``torch.long``.
    """

    def _ensure_batch_tensor(x) -> torch.Tensor:
        """Convert `x` into a 2D tensor [B, L]; if ragged, pad with `pad_token_id` first."""
        # Ragged cases before converting to Tensor
        if (
            isinstance(x, (list, tuple))
            and x
            and isinstance(x[0], (list, tuple, np.ndarray))
        ):
            seqs = [list(s) for s in x]
            max_len = max((len(s) for s in seqs), default=0)
            padded = [(s + [pad_token_id] * (max_len - len(s)))[:max_len] for s in seqs]
            return torch.as_tensor(padded)
        if isinstance(x, np.ndarray) and x.dtype == object:
            seqs = x.tolist()
            max_len = max((len(s) for s in seqs), default=0)
            padded = [
                (list(s) + [pad_token_id] * (max_len - len(s)))[:max_len] for s in seqs
            ]
            return torch.as_tensor(padded)

        t = x if torch.is_tensor(x) else torch.as_tensor(x)
        if t.dim() == 1:  # [L] -> [1, L]
            t = t.unsqueeze(0)
        return t

    input_ids, attention_mask = None, None
    # 1) Mapping (e.g., BatchEncoding)
    if isinstance(tok_output, Mapping) and ("input_ids" in tok_output):
        input_ids = tok_output["input_ids"]
        attention_mask = tok_output.get("attention_mask", None)
    # 2) Tuple/List: (ids, mask) or List[Encoding] or List[List[int]]
    elif isinstance(tok_output, (tuple, list)) and len(tok_output) > 0:
        first_item = tok_output[0]
        # 2a) explicit (ids, mask)
        if len(tok_output) == 2 and not hasattr(first_item, "input_ids"):
            input_ids, attention_mask = tok_output[0], tok_output[1]
        else:
            # 2b) list[Encoding]
            if hasattr(first_item, "input_ids"):
                input_ids = [enc.input_ids for enc in tok_output]
                attention_mask = [
                    getattr(enc, "attention_mask", [1] * len(enc.input_ids))
                    for enc in tok_output
                ]
            else:
                # 2c) treat as list[list[int]]
                input_ids, attention_mask = tok_output, None
    # 3) Single Encoding / EncodingFast
    elif hasattr(tok_output, "input_ids"):
        input_ids = tok_output.input_ids
        attention_mask = getattr(tok_output, "attention_mask", None)
    # 4) Fallback: ids only
    else:
        input_ids = tok_output
        attention_mask = None
    # fit to [batch_size, seq_len]
    input_ids = _ensure_batch_tensor(input_ids)
    if attention_mask is None:
        attention_mask = (input_ids != pad_token_id).to(torch.long)
    else:
        attention_mask = _ensure_batch_tensor(attention_mask).to(torch.long)
        # Shape alignment with input_ids
        batch_size, seq_len = input_ids.shape
        if attention_mask.shape != (batch_size, seq_len):
            if attention_mask.shape[0] != batch_size:
                attention_mask = (input_ids != pad_token_id).to(torch.long)  # rebuild
            else:
                if attention_mask.shape[1] < seq_len:
                    pad_cols = seq_len - attention_mask.shape[1]
                    pad_zeros = torch.zeros(
                        (batch_size, pad_cols), dtype=attention_mask.dtype
                    )
                    attention_mask = torch.cat([attention_mask, pad_zeros], dim=1)
                elif attention_mask.shape[1] > seq_len:
                    attention_mask = attention_mask[:, :seq_len]
    input_ids = input_ids.to(torch.long)
    assert (
        input_ids.dim() == 2
        and attention_mask.dim() == 2
        and input_ids.size() == attention_mask.size()
    ), f"Tokenizer output must be [B,L]; got ids {tuple(input_ids.size())}, mask {tuple(attention_mask.size())}"

    return input_ids, attention_mask


def tokenize_merged_cols(
    df: DataFrame,
    col_types: dict,
    tokenizer_config: "TokenizerConfig",
    target_col: Optional[str] = None,
) -> Optional[tuple]:
    r"""Merge all text columns per row and tokenize.

    Args:
        df (DataFrame): Input table.
        col_types (dict): Mapping of column name to :class:`ColType`.
        tokenizer_config (TokenizerConfig): Tokenizer configuration.
        target_col (Optional[str]): Target column excluded from text merge.

    Returns:
        Optional[tuple]: ``(input_ids, attention_mask)`` with shape
        :math:`(B, L)` if text columns exist; otherwise ``None``.
    """

    text_cols = [
        c for c, t in col_types.items() if t == ColType.TEXT and c != target_col
    ]
    if not text_cols:
        return None

    values_df = df[text_cols].copy()
    values_df = values_df.astype("string")
    values_df = values_df.apply(lambda s: s.str.strip())
    values_df = values_df.replace("", pandas.NA)
    valid_mask = values_df.notna()

    # build per-column segments vectorized
    if tokenizer_config.include_colname:
        name_value_sep = tokenizer_config.name_value_sep
        seg_cols = {}
        for col in text_cols:
            # Use object dtype to avoid NumPy 2.x DTypePromotionError when mixing str and NaN
            s = values_df[col]
            seg = f"{col}{name_value_sep}" + s
            seg = seg.where(valid_mask[col], other=pandas.NA)
            seg_cols[col] = seg
        df_seg = pandas.DataFrame(seg_cols, index=values_df.index)
    else:
        df_seg = values_df.where(valid_mask, other=pandas.NA).astype("string")
    # row-wise merge of non-empty segments
    segment_sep = tokenizer_config.segment_sep
    col_list = df_seg.apply(
        lambda r: segment_sep.join(r.dropna().tolist()), axis=1
    ).tolist()

    input_ids, attention_mask = tokenize_strings(
        col_list,
        tokenizer_config.tokenizer,
        tokenizer_config.pad_token_id,
        standardize_tokenizer_output,
        tokenizer_config.batch_size,
    )
    return input_ids, attention_mask


def save_column_name_tokens(
    col_types: dict,
    tokenizer: Callable,
    pad_token_id: int,
    standardize_func: Callable,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    r"""Tokenize all column names once.

    Args:
        col_types (dict): Mapping of column names to :class:`ColType`.
        tokenizer (Callable): Tokenizer callable.
        pad_token_id (int): Padding token ID.
        standardize_func (Callable): Function that normalizes tokenizer output.

    Returns:
        dict[str, tuple[torch.Tensor, torch.Tensor]]: Mapping from column name
        to token ids and attention mask, each with shape :math:`(L,)`.
    """
    column_names = list(col_types.keys())
    # [C, L], [C, L]
    input_ids, attention_mask = standardize_func(tokenizer(column_names), pad_token_id)

    colname_token_ids = {}
    for i, name in enumerate(column_names):
        colname_token_ids[name] = (
            input_ids[i].clone(),
            attention_mask[i].clone(),
        )  # [L], [L]

    return colname_token_ids
