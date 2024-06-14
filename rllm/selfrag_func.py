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
from langchain_core.output_parsers import StrOutputParser, JsonOutputParsers
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import VectorStoreRetriever 
from langchain_community.embeddings import HuggingFaceEmbeddings, GPT4AllEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings.base import Embeddings
from rllm.utils import get_llm_chat_cost
from sentence_transformers import SentenceTransformer
from typing import List, Literal
from typing_extensions import TypedDict
import pandas as pd
import os
import glob

def self_rag(movie_name: str, prompt: str, retriever: VectorStoreRetriever, **kwargs):
    """
    wrap LangGraph-based pipeline as a whole. Input is a movie name or other information, output is a generated answer.
    """
    llm_model_path = "/home/qinghua_mao/work/rllm/gemma-2b-it-GGUF/gemma-2b-it-q8_0.gguf"
    embed_path = "/home/qinghua_mao/work/rllm/all-MiniLM-L6-v2"

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

    llm = LlamaCpp(
        model_path=llm_model_path,
        streaming=False,
        n_gpu_layers=33,
        verbose=False,
        temperature=0.2,
        n_ctx=2048,
        stop=["\n"],
    )

    if kwargs:
        Title = kwargs['Title']
        Director = kwargs['Director']
        Year = kwargs['Year']
        Genre = kwargs['Genre']
        Cast = kwargs['Cast']
        Runtime = kwargs['Runtime']
        Languages = kwargs['Languages']
        Certificate = kwargs['Certificate']
        Plot = kwargs['Plot']

    ### Retrieval Grader

    retrieval_prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a movie's name. \n 
        Here is the retrieved document: \n\n {document} \n\n
        Here is the user question: {movie_name} \n
        If the document contains keywords related to the movie's name, grade it as relevant. \n
        It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
        Please give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the movie's name. \n
        Provide the binary score as a JSON with a single key 'score' and no premable or explanation.""",
        input_variables=["question", "document"],
    )

    retrieval_grader = retrieval_prompt | llm_json | JsonOutputParser()
    docs = retriever.invoke(movie_name)
    doc_txt = docs[1].page_content
    # docs = get_wiki_info(question)
    print('related', doc_txt)
    print(retrieval_grader.invoke({"movie_name": movie_name, "document": doc_txt}))


    ### Generate

    from langchain import hub
    from langchain_core.output_parsers import StrOutputParser

    # Prompt
    # Construct prompt
    if prompt == 'title':
        prompt_title = """Q: Now I have a movie name: {movie_name}. Here is the retrieved information: {document}. What's the genres it may belong to? 
        Note: 
        1. Give the answer as following format:
        movie_name:: genre_1, genre_2..., genre_n
        2. The answer must only be chosen from followings:'Documentary', 'Adventure', 'Comedy', 'Horror', 'War', 'Sci-Fi', 'Drama', 'Mystery', 'Western', 'Action', "Children's", 'Musical', 'Thriller', 'Crime', 'Film-Noir', 'Romance', 'Animation', 'Fantasy'
        3. Don't saying anything else.
        A: 
        """
        prompt_template = PromptTemplate(
        input_variables=["movie_name", "document"], template=prompt_title
        )

    else:
        prompt_all = """Q: Now I have a movie description: The movie titled '{Title}' is directed by {Director} and was released in {Year}. The genre of this movie is {Genre}, with main cast including {Cast}. It has a runtime of {Runtime} and languages used including {Languages}, with a Certificate rating of {Certificate}. The plot summary is as follows: {Plot}. Here is the retrieved information: {document}. What's the genres it may belong to? 
        Note: 
        1. Give the answer as following format:
        movie_name:: genre_1, genre_2..., genre_n
        2. The answer must only be chosen from followings:'Documentary', 'Adventure', 'Comedy', 'Horror', 'War', 'Sci-Fi', 'Drama', 'Mystery', 'Western', 'Action', "Children's", 'Musical', 'Thriller', 'Crime', 'Film-Noir', 'Romance', 'Animation', 'Fantasy'
        3. Don't saying anything else.
        A: """
        prompt_template = PromptTemplate(
            input_variables=["Title", "Director", "Year", "Genre", "Cast", "Runtime", "Languages", "Certificate", "Plot", "document"], 
            template=prompt_all
        )

    # Post-processing
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Chain
    rag_chain = prompt_template | llm | StrOutputParser()

    # Run
    if prompt == 'title':
        generation = rag_chain.invoke({"document": docs, "movie_name": movie_name})
    else:
        generation = rag_chain.invoke({"document": docs, "Title": Title, "Director": Director, "Year": Year, "Genre": Genre, "Cast": Cast, "Runtime": Runtime, "Languages": Languages, "Certificate": Certificate, "Plot": Plot})
    print(generation)

    ### Hallucination Grader

    # Prompt
    hallu_prompt = PromptTemplate(
        template="""You are a grader assessing whether an answer is grounded in / supported by a set of facts. \n 
        Here are the facts:
        \n ------- \n
        {document} 
        \n ------- \n
        Here is the answer: {generation}
        Give a binary score 'yes' or 'no' score to indicate whether the answer is grounded in / supported by a set of facts. \n
        Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.""",
        input_variables=["generation", "document"],
    )

    hallucination_grader = hallu_prompt | llm_json | JsonOutputParser()
    hallucination_grader.invoke({"document": docs, "generation": generation})

    ### Answer Grader

    # Prompt
    answer_prompt = PromptTemplate(
        template="""You are a grader assessing whether an answer is useful to predict a movie's genre. \n 
        Here is the answer:
        \n ------- \n
        {generation} 
        \n ------- \n
        Here is the question: {movie_name}
        Give a binary score 'yes' or 'no' to indicate whether the answer is useful to predict a movie's genre. \n
        Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.""",
        input_variables=["generation", "movie_name"],
    )

    answer_grader = answer_prompt | llm_json | JsonOutputParser()
    answer_grader.invoke({"movie_name": , "generation": generation})


    ### Question Re-writer

    # Prompt
    re_write_prompt = PromptTemplate(
        template="""You a movie artist that converts a movie's name to a more comprehensive description that is optimized \n 
        for vectorstore retrieval. Look at the initial and formulate an improved interpretation. \n
        Here is the initial movie name: \n\n {movie_name}. Improved name with no preamble: \n """,
        input_variables=["generation", "movie_name"],
    )

    question_rewriter = re_write_prompt | llm | StrOutputParser()
    question_rewriter.invoke({"movie_name": movie_name})


    class GraphState(TypedDict):
        """
        Represents the state of our graph.

        Attributes:
            question: question
            generation: LLM generation
            documents: list of documents
        """

        movie_name: str
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
        question = state["movie_name"]

        # Retrieval
        documents = retriever.get_relevant_documents(movie_name)
        return {"documents": documents, "movie_name": movie_name}


    def generate(state):
        """
        Generate answer

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation, that contains LLM generation
        """
        print("---GENERATE---")
        movie_name = state["movie_name"]
        documents = state["documents"]

        # RAG generation
        generation = rag_chain.invoke({"document": documents, "movie_name": movie_name})
        prompt_cost = get_llm_chat_cost(prompt_template.invoke({"document": documents, "movie_name": movie_name}).text, 'input')
        return {"documents": documents, "movie_name": movie_name, "generation": generation, "cost": prompt_cost}


    def grade_documents(state):
        """
        Determines whether the retrieved documents are relevant to the question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates documents key with only filtered relevant documents
        """

        print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        movie_name = state["movie_name"]
        documents = state["documents"]

        # Score each doc
        filtered_docs = []
        for d in documents:
            score = retrieval_grader.invoke(
                {"movie_name": question, "document": d.page_content}
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
        movie_name = state["movie_name"]
        documents = state["documents"]

        # Re-write question
        better_question = question_rewriter.invoke({"movie_name": movie_name})
        return {"documents": documents, "movie_name": better_question}


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
        movie_name = state["movie_name"]
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
        movie_name = state["movie_name"]
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
            score = answer_grader.invoke({"movie_name": movie_name, "generation": generation})
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
    if prompt == 'title':
        inputs = {"movie_name": movie_name}
    else:
        inputs = {"Title": Title, "Director": Director, "Year": Year, "Genre": Genre, "Cast": Cast, "Runtime": Runtime, "Languages": Languages, "Certificate": Certificate, "Plot": Plot}
    for output in app.stream(inputs):
        for key, value in output.items():
            # Node
            pprint(f"Node '{key}':")
            # Optional: print full state at each node
            # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
        pprint("\n---\n")

    # Final generation
    pprint(value["generation"])