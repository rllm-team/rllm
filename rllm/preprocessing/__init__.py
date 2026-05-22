from typing import TYPE_CHECKING

from rllm.utils.lazy_imports import define_lazy_imports

if TYPE_CHECKING:
    from rllm.preprocessing.fillna import (
        FillNAConfig,
        fillna_by_coltype,
    )
    from rllm.preprocessing.text_tokenize import (
        TokenizerConfig,
        process_tokenized_column,
        tokenize_strings,
        standardize_tokenizer_output,
        tokenize_merged_cols,
        save_column_name_tokens,
    )
    from rllm.preprocessing.word_embedding import (
        TextEmbedderConfig,
        embed_text_column,
    )
    from rllm.preprocessing.data_clean import to_numeric_by_column
    from rllm.preprocessing.timestamp import TimestampPreprocessor
    from rllm.preprocessing.df_to_tensor import df_to_tensor

_LAZY_MODULES = {
    "rllm.preprocessing.fillna": (
        "FillNAConfig",
        "fillna_by_coltype",
    ),
    "rllm.preprocessing.text_tokenize": (
        "TokenizerConfig",
        "process_tokenized_column",
        "tokenize_strings",
        "standardize_tokenizer_output",
        "tokenize_merged_cols",
        "save_column_name_tokens",
    ),
    "rllm.preprocessing.word_embedding": (
        "TextEmbedderConfig",
        "embed_text_column",
    ),
    "rllm.preprocessing.data_clean": ("to_numeric_by_column",),
    "rllm.preprocessing.timestamp": ("TimestampPreprocessor",),
    "rllm.preprocessing.df_to_tensor": ("df_to_tensor",),
}

__all__, __getattr__, __dir__ = define_lazy_imports(
    __name__,
    globals(),
    _LAZY_MODULES,
)
