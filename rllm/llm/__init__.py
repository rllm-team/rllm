from importlib import import_module
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from rllm.llm.predictor import Predictor
    from rllm.llm.enhancer import Enhancer
    from rllm.llm.prompt.base import (
        PromptTemplate,
        ChatPromptTemplate,
    )
    from rllm.llm.llm_module.langchain_llm import LangChainLLM
    from rllm.llm.llm_module.retrieval.retriever import SingleTableRetriever
    from rllm.llm.llm_module.retrieval.retrieval_llm import LLMWithRetriever
    from rllm.llm.llm_module.featllm.feat_engineer import FeatLLMEngineer
    from rllm.llm.llm_module.featllm.feat_llm import FeatLLM
    from rllm.llm.llm_module.finetune.finetuner import (
        FinetuneConfig,
        Seq2SeqFinetuner,
    )
    from rllm.llm.parser.base import BaseOutputParser

_LAZY_MODULES = {
    "rllm.llm.predictor": (
        "Predictor",
    ),
    "rllm.llm.enhancer": (
        "Enhancer",
    ),
    "rllm.llm.prompt.base": (
        "PromptTemplate",
        "ChatPromptTemplate",
    ),
    "rllm.llm.llm_module.langchain_llm": (
        "LangChainLLM",
    ),
    "rllm.llm.llm_module.retrieval.retriever": (
        "SingleTableRetriever",
    ),
    "rllm.llm.llm_module.retrieval.retrieval_llm": (
        "LLMWithRetriever",
    ),
    "rllm.llm.llm_module.featllm.feat_engineer": (
        "FeatLLMEngineer",
    ),
    "rllm.llm.llm_module.featllm.feat_llm": (
        "FeatLLM",
    ),
    "rllm.llm.llm_module.finetune.finetuner": (
        "FinetuneConfig",
        "Seq2SeqFinetuner",
    ),
    "rllm.llm.parser.base": (
        "BaseOutputParser",
    ),
}

__all__ = [
    name
    for names in _LAZY_MODULES.values()
    for name in names
]

_LAZY_ATTRS = {
    name: module_name
    for module_name, names in _LAZY_MODULES.items()
    for name in names
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
