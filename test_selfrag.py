from langchain import hub
from langchain_community.llms import LlamaCpp
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.schema import BaseOutputParser

import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader, CSVLoader
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import HuggingFaceEmbeddings, GPT4AllEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer
from typing import List
import pandas as pd
import os
import glob

llm_model_path = "/home/qinghua_mao/work/rllm/gemma-2b-it-GGUF/gemma-2b-it-q8_0.gguf"
embed_path = "/home/qinghua_mao/work/rllm/all-MiniLM-L6-v2"

local_llm = "mistral"

# embedding class for sentence transformers
class CustomEmbeddings(Embeddings):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        return [self.model.encode(d).tolist() for d in documents]

    def embed_query(self, query: str) -> List[float]:
        return self.model.encode([query])[0].tolist()


class ParquetLoader(BaseLoader):
    def __init__(self, file_path: str):
        super(ParquetLoader, self).__init__()
        self.file_path = file_path
    def load(self) -> List[str]:
        # 使用pandas读取.parquet文件
        df = pd.read_parquet(self.file_path)
        
        # 假设.parquet文件中有一个名为'text'的列，包含了需要加载的文本数据
        # 将这些文本数据转换为字符串列表
        contents = df['text'].tolist()
        idx = df['_id'].tolist()
        urls = df['url'].tolist()
        titles = df['title'].tolist()


        documents = list()
        for content, _id, url, title in zip(contents, idx, urls, titles):
            metadata = {'_id': _id, 'url': url, 'title': title}
            documents.append(Document(page_content=content, metadata=metadata))
        
        return documents


# Load, chunk and index the contents of the blog.
def init_loader(url):
    loader = WebBaseLoader(
        web_paths=(url, ),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )
    return loader

def get_wiki_info(movie_name):
    try:
        summary = wikipedia.summary(movie_name, sentences=2)
    except wikipedia.exceptions.DisambiguationError as e:
        summary = wikipedia.summary(e.options[0], sentences=2)
    except wikipedia.exceptions.PageError:
        summary = "No additional information available."
    return summary

# urls = [
#     "https://lilianweng.github.io/posts/2023-06-23-agent/",
#     "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
#     "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
# ]

urls = glob.glob("/home/qinghua_mao/work/rllm/wikipedia-en/data/*.parquet")  # wiki-en metadata needs to decode into string. wikipedia-en needs build a dict for metadata

# loader = CSVLoader(file_path="/home/qinghua_mao/work/rllm/wikipedia_movies/wiki_movie_plots_deduped_with_summaries.csv")

# docs = loader.load()

docs = [ParquetLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=250, chunk_overlap=0)
doc_splits = text_splitter.split_documents(docs_list)

model_name = "all-MiniLM-L6-v2.gguf2.f16.gguf"
gpt4all_kwargs = {'allow_download': 'True'}
embeddings = GPT4AllEmbeddings(
    model_name = model_name,
    gpt4all_kwargs = gpt4all_kwargs
)
vectorstore = Chroma.from_documents(
    persist_directory="/home/qinghua_mao/work/rllm/chroma_wiki",
    documents=doc_splits,
    collection_name="rag-chroma", 
    embedding=embeddings)
# vectorstore = Chroma(persist_directory="/home/qinghua_mao/work/rllm/chroma", collection_name="rag-chroma", embedding_function=embeddings)

# retrieve and generate using the relevant snippets of the blog
retriever = vectorstore.as_retriever()

### Retrieval Grader

from typing import Literal
from langchain_core.output_parsers import JsonOutputParser
# load model
llm_json = LlamaCpp(
    model_path=llm_model_path,
    streaming=False,
    n_gpu_layers=33,
    verbose=False,
    temperature=0.2,
    n_ctx=1024,
    stop=["\n"],
    grammar_path="/home/qinghua_mao/work/rllm/json.gbnf",
)

# llm_json = ChatOllama(model=local_llm, format="json", temperature=0)

### Retrieval Grader

retrieval_prompt = PromptTemplate(
    template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
    Here is the retrieved document: \n\n {document} \n\n
    Here is the user question: {question} \n
    If the document contains keywords related to the user question, grade it as relevant. \n
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Please give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
    Provide the binary score as a JSON with a single key 'score' and no premable or explanation.""",
    input_variables=["question", "document"],
)

retrieval_grader = retrieval_prompt | llm_json | JsonOutputParser()
question = "comedy movies"
docs = retriever.invoke(question)
doc_txt = docs[1].page_content
# docs = get_wiki_info(question)
print('related', doc_txt)
print(retrieval_grader.invoke({"question": question, "document": docs}))

### Generate

from langchain import hub
from langchain_core.output_parsers import StrOutputParser

# Prompt
prompt = hub.pull("rlm/rag-prompt")

# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# LLM
llm = LlamaCpp(
    model_path=llm_model_path,
    streaming=False,
    n_gpu_layers=33,
    verbose=False,
    temperature=0.2,
    n_ctx=2048,
    stop=["\n"],
)
# llm = ChatOllama(model=local_llm, temperature=0)

# Chain
rag_chain = prompt | llm | StrOutputParser()

# Run
generation = rag_chain.invoke({"context": docs, "question": question})
print(generation)

### Hallucination Grader

# Prompt
hallu_prompt = PromptTemplate(
    template="""You are a grader assessing whether an answer is grounded in / supported by a set of facts. \n 
    Here are the facts:
    \n ------- \n
    {documents} 
    \n ------- \n
    Here is the answer: {generation}
    Give a binary score 'yes' or 'no' score to indicate whether the answer is grounded in / supported by a set of facts. \n
    Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.""",
    input_variables=["generation", "documents"],
)

hallucination_grader = hallu_prompt | llm_json | JsonOutputParser()
hallucination_grader.invoke({"documents": docs, "generation": generation})

### Answer Grader

# Prompt
answer_prompt = PromptTemplate(
    template="""You are a grader assessing whether an answer is useful to resolve a question. \n 
    Here is the answer:
    \n ------- \n
    {generation} 
    \n ------- \n
    Here is the question: {question}
    Give a binary score 'yes' or 'no' to indicate whether the answer is useful to resolve a question. \n
    Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.""",
    input_variables=["generation", "question"],
)

answer_grader = answer_prompt | llm_json | JsonOutputParser()
answer_grader.invoke({"question": question, "generation": generation})


### Question Re-writer

# Prompt
re_write_prompt = PromptTemplate(
    template="""You a question re-writer that converts an input question to a better version that is optimized \n 
     for vectorstore retrieval. Look at the initial and formulate an improved question. \n
     Here is the initial question: \n\n {question}. Improved question with no preamble: \n """,
    input_variables=["generation", "question"],
)

question_rewriter = re_write_prompt | llm | StrOutputParser()
question_rewriter.invoke({"question": question})

from typing_extensions import TypedDict
from typing import List


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
    """

    question: str
    generation: str
    documents: List[str]


### Nodes

from langchain.schema import Document


def retrieve(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]

    # Retrieval
    documents = retriever.get_relevant_documents(question)
    return {"documents": documents, "question": question}


def generate(state):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]

    # RAG generation
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}


def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    # Score each doc
    filtered_docs = []
    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score["score"]
        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            continue
    return {"documents": filtered_docs, "question": question}


def transform_query(state):
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """

    print("---TRANSFORM QUERY---")
    question = state["question"]
    documents = state["documents"]

    # Re-write question
    better_question = question_rewriter.invoke({"question": question})
    return {"documents": documents, "question": better_question}


### Edges


def decide_to_generate(state):
    """
    Determines whether to generate an answer, or re-generate a question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ASSESS GRADED DOCUMENTS---")
    question = state["question"]
    filtered_documents = state["documents"]

    if not filtered_documents:
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print(
            "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---"
        )
        return "transform_query"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"


def grade_generation_v_documents_and_question(state):
    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )
    grade = score["score"]

    # Check hallucination
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # Check question-answering
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke({"question": question, "generation": generation})
        grade = score["score"]
        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        pprint("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"

from langgraph.graph import END, StateGraph

workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("retrieve", retrieve)  # retrieve
workflow.add_node("grade_documents", grade_documents)  # grade documents
workflow.add_node("generate", generate)  # generatae
workflow.add_node("transform_query", transform_query)  # transform_query

# Build graph
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
    },
)
workflow.add_edge("transform_query", "retrieve")
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",
        "useful": END,
        "not useful": "transform_query",
    },
)

# Compile
app = workflow.compile()

from pprint import pprint

# Run
inputs = {"question": "could you please provide two classic comedy movies?"}
for output in app.stream(inputs):
    for key, value in output.items():
        # Node
        pprint(f"Node '{key}':")
        # Optional: print full state at each node
        # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
    pprint("\n---\n")

# Final generation
pprint(value["generation"])