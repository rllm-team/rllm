from importlib import import_module
from typing import TYPE_CHECKING

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

__all__ = [name for names in _LAZY_MODULES.values() for name in names]

_LAZY_ATTRS = {
    name: module_name for module_name, names in _LAZY_MODULES.items() for name in names
}


def __getattr__(name):
    if name in _LAZY_ATTRS:
        module = import_module(_LAZY_ATTRS[name])
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
