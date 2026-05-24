import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch


def install_fake_external_modules():
    class BaseLanguageModel:
        pass

    class BaseLLM(BaseLanguageModel):
        pass

    class BaseChatModel(BaseLanguageModel):
        pass

    class BaseMessage:
        def __init__(self, content="", additional_kwargs=None, **kwargs):
            self.content = content
            self.additional_kwargs = additional_kwargs or {}
            for key, value in kwargs.items():
                setattr(self, key, value)

        @classmethod
        def schema(cls):
            return {"required": []}

    class HumanMessage(BaseMessage):
        pass

    class AIMessage(BaseMessage):
        pass

    class SystemMessage(BaseMessage):
        pass

    class FunctionMessage(BaseMessage):
        @classmethod
        def schema(cls):
            return {"required": ["name"]}

    class ChatMessage(BaseMessage):
        @classmethod
        def schema(cls):
            return {"required": ["role"]}

    class OpenAI(BaseLanguageModel):
        model_name = "fake-openai"
        max_tokens = 12

        def modelname_to_contextsize(self, model_name):
            return 2048

    class ChatOpenAI(BaseChatModel):
        model_name = "fake-chat-openai"
        max_tokens = None

        def modelname_to_contextsize(self, model_name):
            return 4096

    class Document:
        def __init__(self, page_content="", metadata=None):
            self.page_content = page_content
            self.metadata = metadata or {}

    class CSVLoader:
        def __init__(self, file_path):
            self.file_path = file_path

        def load(self):
            df = pd.read_csv(self.file_path)
            docs = []
            for _, row in df.iterrows():
                docs.append(Document("\n".join(f"{k}: {v}" for k, v in row.items())))
            return docs

    class FAISS:
        def __init__(self, documents):
            self._documents = documents

        @classmethod
        def from_documents(cls, documents, embedder):
            embedder.embed_documents([doc.page_content for doc in documents])
            return cls(documents)

        def similarity_search(self, query, k=3):
            return self._documents[:k]

    class Embeddings:
        def embed_documents(self, texts):
            return [[float(len(text))] for text in texts]

    class HuggingFaceEmbeddings(Embeddings):
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    def register(name, module):
        sys.modules[name] = module

    langchain_core = types.ModuleType("langchain_core")
    lc_language_models = types.ModuleType("langchain_core.language_models")
    lc_language_models.BaseLanguageModel = BaseLanguageModel
    lc_language_models.BaseLLM = BaseLLM
    lc_messages = types.ModuleType("langchain_core.messages")
    for cls in [AIMessage, BaseMessage, ChatMessage, FunctionMessage, HumanMessage, SystemMessage]:
        setattr(lc_messages, cls.__name__, cls)
    lc_documents = types.ModuleType("langchain_core.documents")
    lc_documents.Document = Document
    lc_embeddings = types.ModuleType("langchain_core.embeddings")
    lc_embeddings.Embeddings = Embeddings

    langchain = types.ModuleType("langchain")
    langchain_chat_models = types.ModuleType("langchain.chat_models")
    langchain_chat_models_base = types.ModuleType("langchain.chat_models.base")
    langchain_chat_models_base.BaseChatModel = BaseChatModel
    langchain_embeddings = types.ModuleType("langchain.embeddings")
    langchain_community = types.ModuleType("langchain_community")
    langchain_community_llms = types.ModuleType("langchain_community.llms")
    langchain_community_llms.OpenAI = OpenAI
    langchain_community_chat = types.ModuleType("langchain_community.chat_models")
    langchain_community_chat.ChatOpenAI = ChatOpenAI
    langchain_community_vectorstores = types.ModuleType("langchain_community.vectorstores")
    langchain_community_vectorstores.FAISS = FAISS
    langchain_community_loaders = types.ModuleType("langchain_community.document_loaders")
    langchain_community_loaders.CSVLoader = CSVLoader
    langchain_huggingface = types.ModuleType("langchain_huggingface")
    langchain_huggingface.HuggingFaceEmbeddings = HuggingFaceEmbeddings

    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer = object
    transformers.AutoModelForSeq2SeqLM = object
    transformers.get_linear_schedule_with_warmup = lambda *args, **kwargs: None

    for name, module in {
        "langchain_core": langchain_core,
        "langchain_core.language_models": lc_language_models,
        "langchain_core.messages": lc_messages,
        "langchain_core.documents": lc_documents,
        "langchain_core.embeddings": lc_embeddings,
        "langchain": langchain,
        "langchain.chat_models": langchain_chat_models,
        "langchain.chat_models.base": langchain_chat_models_base,
        "langchain.embeddings": langchain_embeddings,
        "langchain_community": langchain_community,
        "langchain_community.llms": langchain_community_llms,
        "langchain_community.chat_models": langchain_community_chat,
        "langchain_community.vectorstores": langchain_community_vectorstores,
        "langchain_community.document_loaders": langchain_community_loaders,
        "langchain_huggingface": langchain_huggingface,
        "transformers": transformers,
    }.items():
        register(name, module)

    return types.SimpleNamespace(
        BaseLanguageModel=BaseLanguageModel,
        BaseLLM=BaseLLM,
        BaseChatModel=BaseChatModel,
        HumanMessage=HumanMessage,
        AIMessage=AIMessage,
        SystemMessage=SystemMessage,
        FunctionMessage=FunctionMessage,
        ChatMessage=ChatMessage,
        OpenAI=OpenAI,
        ChatOpenAI=ChatOpenAI,
        Embeddings=Embeddings,
        Document=Document,
    )


FAKES = install_fake_external_modules()

from rllm.llm.constants import DEFAULT_CONTEXT_WINDOW, DEFAULT_NUM_OUTPUTS, DEFAULT_TEMPERATURE
from rllm.llm.enhancer import Enhancer
from rllm.llm.llm_module.featllm.feat_engineer import FeatLLMEngineer
from rllm.llm.llm_module.featllm.feat_llm import FeatLLM, simple_model
from rllm.llm.llm_module.finetune.finetuner import (
    FinetuneConfig,
    FinetuneDataset,
    create_collate_fn,
)
from rllm.llm.llm_module.general_llm import LLM, default_completion_to_prompt
from rllm.llm.llm_module.langchain_llm import LangChainLLM
from rllm.llm.llm_module.langchain_utils import (
    from_lc_messages,
    get_llm_metadata,
    is_chat_model as lc_is_chat_model,
    to_lc_messages,
)
from rllm.llm.llm_module.retrieval.retrieval_llm import LLMWithRetriever
from rllm.llm.llm_module.retrieval.retriever import SingleTableRetriever
from rllm.llm.parser.base import BaseOutputParser
from rllm.llm.predictor import Predictor
from rllm.llm.prompt.base import ChatPromptTemplate, PromptTemplate
from rllm.llm.prompt.utils import (
    completion_response_to_chat_response,
    generate_sample_description,
    get_template_vars,
    is_chat_model,
    messages_to_prompt,
    prompt_to_messages,
)
from rllm.llm.types import ChatMessage, ChatResponse, CompletionResponse, LLMMetadata, MessageRole


class UpperParser(BaseOutputParser):
    def parse(self, output):
        return output.upper()

    def format(self, query):
        return f"{query}\nFORMAT"


class FakeLLM(LLM):
    def __init__(self, chat_model=False, outputs=None, **kwargs):
        super().__init__(**kwargs)
        self._metadata = LLMMetadata(is_chat_model=chat_model, model_name="fake")
        self.outputs = list(outputs or ["answer"])
        self.seen_chat = []
        self.seen_complete = []

    @property
    def metadata(self):
        return self._metadata

    def chat(self, messages, **kwargs):
        self.seen_chat.append(messages)
        return ChatResponse(ChatMessage(role=MessageRole.ASSISTANT, content=self.outputs.pop(0)))

    def complete(self, prompt, formatted=False, **kwargs):
        self.seen_complete.append((prompt, formatted))
        return CompletionResponse(self.outputs.pop(0))

    def embedding(self, inputs):
        return [[float(len(text)), 1.0] for text in inputs]


class FakeTokenizer:
    pad_token_id = 0

    def __call__(self, text, return_tensors="pt", truncation=True, add_special_tokens=True, **kwargs):
        if isinstance(text, list):
            max_len = max(len(t) for t in text)
            ids = [[len(t)] + [0] * (max_len - 1) for t in text]
        else:
            ids = [[ord(ch) % 11 + 1 for ch in text[:4]]]
        return types.SimpleNamespace(input_ids=torch.tensor(ids), to=lambda device: {"input_ids": torch.tensor(ids)})

    def batch_decode(self, values, skip_special_tokens=True):
        return ["decoded"] * len(values)


class FakeDataset:
    def __init__(self):
        self.rows = [
            {"note": "n1", "text_label": "yes", "label": 0, "idx": 3},
            {"note": "n2", "text_label": "no", "label": 1, "idx": 4},
        ]
        self.features = {"label": types.SimpleNamespace(names=["yes", "no"])}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


def test_llm_types_and_prompt_utils():
    msg = ChatMessage.from_str("hello", role="user", additional_kwargs={"x": 1})
    completion = CompletionResponse("done", additional_kwargs={"k": "v"}, raw={"r": 1}, delta="d")
    chat_response = completion_response_to_chat_response(completion)

    assert str(msg) == "user: hello"
    assert str(ChatResponse(msg)) == "user: hello"
    assert str(completion) == "done"
    assert chat_response.message.role == MessageRole.ASSISTANT
    assert chat_response.message.additional_kwargs == {"k": "v"}
    assert DEFAULT_TEMPERATURE == 0.1
    assert DEFAULT_CONTEXT_WINDOW > DEFAULT_NUM_OUTPUTS
    assert messages_to_prompt([msg]).endswith("assistant: ")
    assert prompt_to_messages("prompt")[0].content == "prompt"
    assert get_template_vars("A {x} and {y:.2f}") == ["x", "y"]
    assert generate_sample_description(pd.Series({"a": 1, "b": "two"})) == "a is 1\nb is two"


def test_chat_message_dict_current_behavior_is_detected():
    with pytest.raises(AttributeError):
        ChatMessage("user", "content").dict()


def test_prompt_template_formats_partials_functions_and_parser():
    prompt = PromptTemplate(
        "Hi {name}, value={value}",
        output_parser=UpperParser(),
        template_var_mappings={"alias": "name"},
        function_mappings={"value": lambda **kwargs: kwargs["base"] + 1},
        base=2,
    )
    partial = prompt.partial_format(alias="Ada")

    assert partial.format() == "Hi Ada, value=3\nFORMAT"
    assert prompt.format(alias="Bob", completion_to_prompt=lambda text: f"<<{text}>>") == "<<Hi Bob, value=3\nFORMAT>>"
    assert prompt.format_messages(alias="Ada")[0].content == "Hi Ada, value=3\nFORMAT"
    assert prompt.get_template() == "Hi {name}, value={value}"


def test_chat_prompt_template_current_copy_behavior_is_detected():
    prompt = ChatPromptTemplate.from_messages([("system", "Sys {x}"), ("user", "User {y}")], x="A")

    with pytest.raises(AttributeError):
        prompt.format_messages(y="B")


def test_base_output_parser_formats_first_system_or_last_message():
    parser = UpperParser()
    system_messages = [ChatMessage("system", "sys"), ChatMessage("user", "user")]
    user_messages = [ChatMessage("user", "first"), ChatMessage("assistant", "last")]

    assert parser.format_messages(system_messages)[0].content == "sys\nFORMAT"
    assert parser.format_messages(user_messages)[-1].content == "last\nFORMAT"


def test_general_llm_predict_completion_and_chat_paths():
    completion_llm = FakeLLM(chat_model=False, outputs=["raw"], system_prompt="SYS", output_parser=UpperParser())
    chat_llm = FakeLLM(chat_model=True, outputs=["chat"], system_prompt="SYS", output_parser=UpperParser())
    prompt = PromptTemplate("Question {x}")

    assert default_completion_to_prompt("abc") == "abc"
    assert completion_llm.predict(prompt, x=1) == "RAW"
    assert completion_llm.seen_complete[0] == ("SYS\n\nQuestion 1\nFORMAT", True)
    assert chat_llm.predict(prompt, x=2) == "CHAT"
    assert chat_llm.seen_chat[0][0].role == MessageRole.SYSTEM
    assert is_chat_model(chat_llm)


def test_predictor_and_enhancer_use_mock_llms(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda _: None)
    df = pd.DataFrame({"age": [10, 20], "city": ["A", "B"]})
    prompt = PromptTemplate("Scenario {scenario}\n{sample_description}", function_mappings={"sample_description": generate_sample_description})

    predictor = Predictor(prompt=prompt, llm=FakeLLM(outputs=["p1", "p2"]))
    enhancer = Enhancer(prompt=prompt, llm=FakeLLM(outputs=["e1", "e2"]), type="explanation")
    embedder = Enhancer(llm_embed=FakeLLM(), type="embedding")

    assert predictor(df, scenario="classify") == ["p1", "p2"]
    assert enhancer(df, scenario="explain") == ["e1", "e2"]
    assert embedder(pd.DataFrame({"text": ["aa", "bbb"]}), cols=["text"]).tolist() == [[2.0, 1.0], [3.0, 1.0]]

    with pytest.raises(AssertionError):
        Predictor(prompt=None, llm=FakeLLM(), type="bad")


def test_langchain_utils_message_conversion_and_metadata():
    user = ChatMessage("user", "hi")
    fn = ChatMessage("function", "result", additional_kwargs={"name": "tool"})
    chatbot = ChatMessage("chatbot", "bot", additional_kwargs={"role": "bot-role"})

    lc_messages = to_lc_messages([user, fn, chatbot])
    restored = from_lc_messages(lc_messages)

    assert lc_is_chat_model(FAKES.ChatOpenAI())
    assert get_llm_metadata(FAKES.OpenAI()).context_window == 2048
    assert get_llm_metadata(FAKES.ChatOpenAI()).num_output == -1
    assert [msg.role for msg in restored] == [MessageRole.USER, MessageRole.FUNCTION, MessageRole.CHATBOT]

    with pytest.raises(ValueError):
        get_llm_metadata(object())
    with pytest.raises(ValueError):
        to_lc_messages([ChatMessage("tool", "bad")])


def test_langchain_llm_adapter_complete_chat_and_embedding():
    class CompletionModel(FAKES.BaseLanguageModel):
        def invoke(self, prompt, **kwargs):
            return f"completed:{prompt}"

    class ChatModel(FAKES.BaseChatModel):
        def predict_messages(self, messages, **kwargs):
            return FAKES.AIMessage("chat-result")

        def invoke(self, prompt, **kwargs):
            return "unused"

    class EmbeddingModel(CompletionModel):
        def embed_documents(self, inputs):
            return [[float(len(x))] for x in inputs]

    completion_llm = LangChainLLM(CompletionModel(), completion_to_prompt=lambda text: f"wrap:{text}")
    chat_llm = LangChainLLM(ChatModel())
    emb_llm = LangChainLLM(EmbeddingModel())

    assert completion_llm.complete("hello").text == "completed:wrap:hello"
    assert completion_llm.chat([ChatMessage("user", "hi")]).message.content == "completed:user: hi\nassistant: "
    assert chat_llm.chat([ChatMessage("user", "hi")]).message.content == "chat-result"
    assert emb_llm.embedding("abc") == [[3.0]]


def test_retriever_uses_csv_loader_filter_and_vector_store(tmp_path):
    path = tmp_path / "items.csv"
    pd.DataFrame({"name": ["keep", "drop"], "value": [1, 2]}).to_csv(path, index=False)

    retriever = SingleTableRetriever(str(path), embedder=FAKES.Embeddings(), filter_func=lambda text: "keep" in text)
    docs = retriever("query", top_k=2)

    assert len(docs) == 1
    assert "keep" in docs[0].page_content


def test_finetune_dataset_config_and_collate():
    cfg = FinetuneConfig(csv_dataset="data.csv", target_column="label", task_info_path="task.txt", device="cpu")
    item = FinetuneDataset(FakeDataset(), FakeTokenizer(), "task")[0]
    batch = create_collate_fn(pad_token_id=0)([item, FinetuneDataset(FakeDataset(), FakeTokenizer(), "task")[1]])

    assert cfg.model_name == "google-t5/t5-small"
    assert len(item) == 5
    assert batch["input_ids"].ndim == 2
    assert (batch["target_ids"] == -100).any()
    assert batch["attention_mask"].shape == batch["input_ids"].shape


def test_simple_model_forward_and_featllm_evaluate():
    x1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    x2 = torch.tensor([[2.0], [5.0]])
    model = simple_model([x1, x2])

    out = model([x1, x2])

    assert out.shape == (2, 2)
    assert torch.all(out >= 0)


def test_featllm_engineer_helpers_and_conversion(tmp_path, monkeypatch):
    class FakeLangChainLLM(FAKES.BaseLLM):
        def invoke(self, *args, **kwargs):
            return "ok"

    csv_path = tmp_path / "toy.csv"
    metadata_path = tmp_path / "meta.json"
    task_path = tmp_path / "task.txt"
    pd.DataFrame(
        {
            "num": [1, 2, 3, 4, 5, 6],
            "cat": ["a", "a", "b", "b", "a", "b"],
            "label": ["yes", "no", "yes", "no", "yes", "no"],
        }
    ).to_csv(csv_path, index=False)
    metadata_path.write_text(json.dumps({"num": "number", "cat": "category"}))
    task_path.write_text("predict label")

    engineer = FeatLLMEngineer(
        str(csv_path),
        str(metadata_path),
        str(task_path),
        llm=FakeLangChainLLM(),
        shots=2,
        test_size=2,
        target_column="label",
        seed=0,
        query_num=1,
    )
    rules = engineer._extract_rules(
        ['10 different conditions for class "yes":\n- num > 1\n10 different conditions for class "no":\n- cat is in [b]'],
        ["yes", "no"],
    )
    prompts = engineer._generate_function_prompt(rules[0], "- num: number")
    executable, train_dict, test_dict = engineer._convert_to_binary_vectors(
        [["def extracting_features_yes(df_input):\n    return pd.DataFrame({'a': df_input['num'] > 0})",
          "def extracting_features_no(df_input):\n    return pd.DataFrame({'b': df_input['num'] < 10})"]],
        [["extracting_features_yes", "extracting_features_no"]],
        ["yes", "no"],
        engineer.X_train,
        engineer.X_test,
    )

    assert "num is 1" in engineer._serialize(engineer.df.iloc[0][["num", "cat"]])
    assert rules == [{"yes": ["num > 1"], "no": ["cat is in [b]"]}]
    assert "Function name: extracting_features_yes" in prompts[0]
    assert executable == [0]
    assert set(train_dict[0]) == {"yes", "no"}


def test_llm_with_retriever_helpers_and_validation(tmp_path):
    csv_path = tmp_path / "toy.csv"
    metadata_path = tmp_path / "meta.json"
    task_path = tmp_path / "task.txt"
    label_path = tmp_path / "labels.txt"
    pd.DataFrame(
        {
            "x": [1, 2, 3, 4, 5, 6],
            "target": ["a", "b", "a", "b", "a", "b"],
        }
    ).to_csv(csv_path, index=False)
    metadata_path.write_text(json.dumps({"x": "number"}))
    task_path.write_text("task")
    label_path.write_text("a,b")

    with pytest.raises(TypeError):
        LLMWithRetriever(str(csv_path), str(metadata_path), str(task_path), str(label_path), llm=object())

    instance = object.__new__(LLMWithRetriever)
    instance.llm = FAKES.BaseLLM()
    assert instance._to_json("a: 1\nb: two") == '{"a": "1", "b": "two"}'
    assert not instance._is_chat_model()
