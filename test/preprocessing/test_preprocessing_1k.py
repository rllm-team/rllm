import math

import numpy as np
import pandas as pd
import pytest
import torch

from rllm.preprocessing._fillna import (
    fillna_binary,
    fillna_by_coltype,
    fillna_categorical,
    fillna_numerical,
    fillna_text,
    fillna_timestamp,
)
from rllm.preprocessing._type_convert import (
    convert_binary,
    dict_to_df,
    encode_categorical,
)
from rllm.preprocessing.data_clean import to_numeric_by_column
from rllm.preprocessing.df_to_tensor import df_to_tensor
from rllm.preprocessing.text_tokenize import (
    TokenizerConfig,
    process_tokenized_column,
    save_column_name_tokens,
    standardize_tokenizer_output,
    tokenize_merged_cols,
    tokenize_strings,
)
from rllm.preprocessing.timestamp import TimestampPreprocessor
from rllm.preprocessing.word_embedding import TextEmbedderConfig, embed_text_column
from rllm.types import ColType


class FixedTokenizer:
    def __call__(self, texts):
        ids = [[len(text), i + 1, 0] for i, text in enumerate(texts)]
        mask = [[1, 1, 0] for _ in texts]
        return {"input_ids": ids, "attention_mask": mask}


class EncodingLike:
    def __init__(self, input_ids, attention_mask=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask


def test_fillna_numerical_strategies_and_errors():
    series = pd.Series([1.0, 2.0, np.nan, 4.0])

    assert fillna_numerical(series, strategy="mean").iloc[2] == pytest.approx(7.0 / 3.0)
    assert fillna_numerical(series, strategy="median").iloc[2] == 2.0
    assert fillna_numerical(series, strategy="constant", fill_value=-9.0).iloc[2] == -9.0

    all_missing = pd.Series([np.nan, np.nan])
    assert fillna_numerical(all_missing, fill_value=3.5).tolist() == [3.5, 3.5]

    with pytest.raises(ValueError):
        fillna_numerical(series, strategy="unknown")


def test_fillna_timestamp_strategies_and_errors():
    dates = pd.Series(pd.to_datetime(["2020-01-01", None, "2020-01-03"]))

    assert fillna_timestamp(dates, strategy="bfill").iloc[1] == pd.Timestamp("2020-01-03")
    assert fillna_timestamp(dates, strategy="constant", fill_value=pd.Timestamp("1999-01-01")).iloc[1] == pd.Timestamp("1999-01-01")

    median_filled = fillna_timestamp(dates, strategy="median")
    assert median_filled.iloc[1] == pd.Timestamp("2020-01-02")

    with pytest.raises(ValueError):
        fillna_timestamp(dates, strategy="unknown")


def test_fillna_by_coltype_dispatches_expected_defaults():
    assert fillna_by_coltype(pd.Series(["A", None]), ColType.CATEGORICAL).tolist() == ["A", -1]
    assert fillna_by_coltype(pd.Series(["hello", None]), ColType.TEXT).tolist() == ["hello", ""]
    assert fillna_by_coltype(pd.Series([1, 1, 0, None]), ColType.BINARY).tolist() == [1, 1, 0, 1]


def test_fillna_individual_column_types():
    dates = pd.Series(pd.to_datetime([None, "2020-01-02", None]))

    assert fillna_categorical(pd.Series(["x", None])).tolist() == ["x", -1]
    assert fillna_text(pd.Series(["x", None])).tolist() == ["x", ""]
    assert fillna_binary(pd.Series([0, 1, 1, None])).tolist() == [0, 1, 1, 1]
    assert fillna_timestamp(dates, strategy="ffill").tolist() == [
        pd.Timestamp("2020-01-02"),
        pd.Timestamp("2020-01-02"),
        pd.Timestamp("2020-01-02"),
    ]


def test_to_numeric_by_column_cleans_common_formats():
    result = to_numeric_by_column(pd.Series(["$1,234.56", "10%", "1.5e2", "bad", None]))

    assert result.iloc[0] == 1234.56
    assert result.iloc[1] == 0.1
    assert result.iloc[2] == 150.0
    assert pd.isna(result.iloc[3])
    assert pd.isna(result.iloc[4])


def test_encode_categorical_preserves_missing_as_minus_one():
    encoded, label_map = encode_categorical(pd.Series(["cat", "dog", "cat", None, "bird", "-1"]))

    assert encoded.tolist()[3] == -1
    assert encoded.tolist()[5] == -1
    assert set(label_map.values()) == {"bird", "cat", "dog"}
    assert encoded.dtype == "int64"


def test_convert_binary_is_case_insensitive_and_defaults_unknown_to_zero():
    result = convert_binary(pd.Series(["yes", "No", "TRUE", "false", "1", "0", "maybe"]))

    assert result.tolist() == [1, 0, 1, 0, 1, 0, 0]


def test_dict_to_df_reconstructs_requested_columns_in_type_order():
    data_dict = {
        ColType.NUMERICAL: torch.tensor([[1.0], [2.0]]),
        ColType.CATEGORICAL: torch.tensor([[0], [1]]),
        ColType.BINARY: torch.tensor([[1], [0]]),
    }

    df = dict_to_df(
        data_dict,
        numerical_columns=["num"],
        categorical_columns=["cat"],
        binary_columns=["bin"],
    )

    assert df.to_dict("list") == {"cat": [0, 1], "num": [1.0, 2.0], "bin": [1, 0]}


def test_timestamp_preprocessor_default_custom_and_invalid_fields():
    default_tensor = TimestampPreprocessor()(pd.Series(["2020-01-02 03:04:05", None]))
    custom_tensor = TimestampPreprocessor(fields=["YEAR", "MONTH"])(pd.Series(["2020-02-01"]))

    assert default_tensor.shape == (2, 7)
    assert default_tensor.dtype == torch.long
    assert default_tensor[0].tolist() == [2020, 0, 1, 3, 3, 4, 5]
    assert default_tensor[1].tolist() == [-1, -1, -1, -1, -1, -1, -1]
    assert custom_tensor.tolist() == [[2020, 1]]

    with pytest.raises(ValueError):
        TimestampPreprocessor(fields=["NOT_A_FIELD"])


def test_standardize_tokenizer_output_handles_supported_shapes():
    ids, mask = standardize_tokenizer_output({"input_ids": [[1, 2], [3]], "attention_mask": [[1], [1, 0]]}, pad_token_id=0)
    assert ids.tolist() == [[1, 2], [3, 0]]
    assert mask.tolist() == [[1, 0], [1, 0]]

    ids, mask = standardize_tokenizer_output([EncodingLike([5, 6]), EncodingLike([7], [1])], pad_token_id=0)
    assert ids.tolist() == [[5, 6], [7, 0]]
    assert mask.tolist() == [[1, 1], [1, 0]]

    ids, mask = standardize_tokenizer_output([9, 8, 0], pad_token_id=0)
    assert ids.tolist() == [[9, 8, 0]]
    assert mask.tolist() == [[1, 1, 0]]


def test_tokenize_strings_batches_and_processes_column_names():
    calls = []

    def tokenizer(texts):
        calls.append(list(texts))
        return [[len(text)] for text in texts]

    ids, mask = tokenize_strings(["a", "bb", "ccc"], tokenizer, 0, standardize_tokenizer_output, batch_size=2)
    assert calls == [["a", "bb"], ["ccc"]]
    assert ids.tolist() == [[1], [2], [3]]
    assert mask.tolist() == [[1], [1], [1]]

    config = TokenizerConfig(tokenizer=FixedTokenizer(), include_colname=True, name_value_sep="=")
    ids, _ = process_tokenized_column(pd.Series(["x", "yy"]), "text", config)
    assert ids[:, 0].tolist() == [6, 7]


def test_tokenize_merged_cols_and_save_column_name_tokens():
    df = pd.DataFrame({"title": [" A ", ""], "body": ["B", None], "num": [1, 2]})
    col_types = {"title": ColType.TEXT, "body": ColType.TEXT, "num": ColType.NUMERICAL}
    config = TokenizerConfig(
        tokenizer=lambda texts: {"input_ids": [[len(text), 0] for text in texts]},
        include_colname=True,
        segment_sep="|",
        name_value_sep=":",
    )

    ids, mask = tokenize_merged_cols(df, col_types, config)
    assert ids.tolist() == [[14, 0], [0, 0]]
    assert mask.tolist() == [[1, 0], [0, 0]]

    col_tokens = save_column_name_tokens(
        col_types,
        lambda names: {"input_ids": [[len(name)] for name in names]},
        0,
        standardize_tokenizer_output,
    )
    assert col_tokens["title"][0].tolist() == [5]
    assert col_tokens["num"][0].tolist() == [3]


def test_df_to_tensor_converts_core_column_types_and_target():
    df = pd.DataFrame(
        {
            "num": ["$1.00", None, "3"],
            "cat": ["A", None, "B"],
            "bin": ["yes", None, "0"],
            "ts": ["2020-01-01 01:02:03", None, "2020-01-03 04:05:06"],
            "target": [0, 1, 0],
        }
    )
    col_types = {
        "num": ColType.NUMERICAL,
        "cat": ColType.CATEGORICAL,
        "bin": ColType.BINARY,
        "ts": ColType.TIMESTAMP,
        "target": ColType.BINARY,
    }

    feat_dict, y = df_to_tensor(df, col_types, target_col="target")

    assert y.tolist() == [0.0, 1.0, 0.0]
    assert feat_dict[ColType.NUMERICAL].shape == (3, 1)
    assert feat_dict[ColType.CATEGORICAL].shape == (3, 1)
    assert feat_dict[ColType.BINARY].shape == (3, 1)
    assert feat_dict[ColType.TIMESTAMP].shape == (3, 1, 7)
    assert not torch.isnan(feat_dict[ColType.NUMERICAL]).any()


def test_df_to_tensor_tokenizes_text_columns_without_randomness():
    df = pd.DataFrame({"text": ["alpha", "beta"]})
    tokenizer_config = TokenizerConfig(
        tokenizer=FixedTokenizer(),
        tokenize_combine=False,
        include_colname=False,
    )

    feat_dict, y = df_to_tensor(df, {"text": ColType.TEXT}, tokenizer_config=tokenizer_config)
    input_ids, attention_mask = feat_dict[ColType.TEXT]

    assert y is None
    assert input_ids.tolist() == [[[5, 1, 0]], [[4, 2, 0]]]
    assert attention_mask.tolist() == [[[1, 1, 0]], [[1, 1, 0]]]


def test_df_to_tensor_can_merge_multiple_text_columns():
    df = pd.DataFrame({"title": ["A", "BB"], "body": ["CCC", "D"]})
    config = TokenizerConfig(
        tokenizer=lambda texts: {"input_ids": [[len(text)] for text in texts]},
        tokenize_combine=True,
        include_colname=False,
        segment_sep=" ",
    )

    feat_dict, _ = df_to_tensor(df, {"title": ColType.TEXT, "body": ColType.TEXT}, tokenizer_config=config)
    ids, mask = feat_dict[ColType.TEXT]

    assert ids.tolist() == [[5], [4]]
    assert mask.tolist() == [[1], [1]]


def test_df_to_tensor_embeds_text_columns():
    df = pd.DataFrame({"text": ["alpha", "beta", "gamma"]})

    def embedder(texts):
        return torch.tensor([[len(text), math.isfinite(len(text))] for text in texts])

    feat_dict, _ = df_to_tensor(
        df,
        {"text": ColType.TEXT},
        text_embedder_config=TextEmbedderConfig(text_embedder=embedder),
    )

    assert feat_dict[ColType.TEXT].shape == (3, 1, 2)
    assert feat_dict[ColType.TEXT][:, 0, 0].tolist() == [5.0, 4.0, 5.0]


def test_embed_text_column_batches_and_df_to_tensor_concat_false():
    calls = []

    def embedder(texts):
        calls.append(list(texts))
        return torch.tensor([[len(text), len(text) + 1] for text in texts])

    embeddings = embed_text_column(
        pd.Series(["a", "bb", "ccc"]),
        TextEmbedderConfig(text_embedder=embedder, batch_size=2),
    )
    assert calls == [["a", "bb"], ["ccc"]]
    assert embeddings.tolist() == [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]

    feat_dict, _ = df_to_tensor(
        pd.DataFrame({"num": [1, 2], "cat": ["x", "y"]}),
        {"num": ColType.NUMERICAL, "cat": ColType.CATEGORICAL},
        concat=False,
        cat_hardcode=False,
    )
    assert isinstance(feat_dict[ColType.NUMERICAL], list)
    assert feat_dict[ColType.CATEGORICAL][0].dtype == torch.float32
