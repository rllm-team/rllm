from typing import Any, List, Callable

import torch
from langchain_core.documents import Document

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import CSVLoader


class SingleTableRetriever:
    """
    A class for semantic retrieval of documents based on a query.

    This class is designed to load a CSV file, optionally filter its content, 
    and create a semantic search index using embeddings. It allows users to 
    retrieve the most relevant documents based on a query.
    """

    def __init__(
        self,
        file_path: str,
        *,
        embedder: Embeddings = None,
        filter_func: Callable[[str], bool] = None,
    ):
        """
        Initialize the single table (CSV format) retriever.

        Args:
            file_path (str): Path to a CSV file containing the text data.
            embedder (Embeddings, optional): Embeddings model to use for encoding text.
                If not provided, a default Sentence-Transformer model is used.
            filter_func (Callable[[str], bool], optional): A function to filter lines 
                from the CSV file. Only rows where the function returns True are kept.
        """
        if embedder is None:
            # Use the default Sentence-Transformer embedding model if no embedder is provided.
            print("-" * 85)
            print("No model given, using default Sentence-Transformer embedder.")
            print("If the loading time is longer than expected, please check your network connection.")
            print("-" * 85)
            self.embedder = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs = {'device': "cuda" if torch.cuda.is_available() else "cpu"},
            )
        else:
            self.embedder = embedder

        # Load documents from the CSV file.
        self.loader = CSVLoader(file_path=file_path)
        self.documents = self.loader.load()

        # Apply the filter function if provided.
        if filter_func:
            self.documents = [
                doc for doc in self.documents if filter_func(doc.page_content)
            ]

        # Create a FAISS index from the loaded documents using the embedder.
        self.documents = FAISS.from_documents(self.documents, self.embedder)

    def retrieve(self, query: str, top_k: int = 3) -> List[Document]:
        """
        Retrieve the most relevant documents based on the query.

        Args:
            query (str): The query text provided by the user.
            top_k (int): The number of top relevant documents to return (default is 3).

        Returns:
            List[Document]: A list of the most relevant documents, sorted by relevance.
        """
        results = self.documents.similarity_search(query, k=top_k)
        return results

    def __call__(
        self, query: str, top_k: int = 3, *args: Any, **kwargs: Any
    ) -> List[Document]:
        """
        Allow the retriever instance to be called directly as a function.

        Args:
            query (str): The query text provided by the user.
            top_k (int): The number of top relevant documents to return (default is 3).
            *args (Any): Additional positional arguments (not used).
            **kwargs (Any): Additional keyword arguments (not used).

        Returns:
            List[Document]: A list of the most relevant documents, sorted by relevance.
        """
        return self.retrieve(query=query, top_k=top_k)


# example
if __name__ == "__main__":
    import os
    import pandas as pd
    # Create a sample CSV file for testing.
    data = {
        "Name": ["Lihua", "Zhaoming", "Kitty"],
        "Age": [15, 27, 18],
        "City": ["Beijing", "Hangzhou", "California"]
    }
    path = "./test.csv"
    df = pd.DataFrame(data)
    df.to_csv(path, index=False)

    # Initialize the retriever with the sample CSV file.
    retriever_default = SingleTableRetriever(file_path=path)

    # Perform a query to retrieve relevant documents.
    query = "Who lives in America?"
    results = retriever_default(query, top_k=1)
    print(results)

    # Clean up the test CSV file.
    os.remove(path)
