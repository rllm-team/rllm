from typing import TYPE_CHECKING

from rllm.utils.lazy_imports import define_lazy_imports


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

__all__, __getattr__, __dir__ = define_lazy_imports(
    __name__,
    globals(),
    _LAZY_MODULES,
)
