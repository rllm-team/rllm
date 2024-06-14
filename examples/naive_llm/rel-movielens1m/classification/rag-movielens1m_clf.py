# -*- coding: ascii -*-

# RAG-enhanced llm pipeline for classification task in rel-movielens1M
# Paper: Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection  https://arxiv.org/abs/2310.11511
# Title only: macro_f1: 0.251, micro_f1: 0.387
# Full info: macro_f1: 0.892, micro_f1: 0.884
# Runtime: Title only: 2990s; Full info: 6757s (on a single 6G GPU)
# Cost: Title only: $0.2722; Full info: $0.5996
# Description: Give llm movie name and limited genres, relevant documents are retrieved from wikipedia database to assist llm in predicting the genres of movies. We introduce self-rag to critique the retrieval and generation with critique tokens.
# Usage: python rag-movielens1m_clf.py --prompt title/all

# Append rllm to search path
import sys
sys.path.append("../../../../")
import time
import argparse

import pandas as pd

from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer

from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.schema import BaseOutputParser

import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rllm.utils import macro_f1_score, micro_f1_score, get_llm_chat_cost
from rllm.selfrag_func import self_rag
##### Parse argument
parser = argparse.ArgumentParser()
parser.add_argument('--prompt', choices=['title', 'all'], 
                    default='title', help='Choose prompt type.')
args = parser.parse_args()

##### Start time
time_start = time.time()

##### Global variables
total_cost = 0
test_path = "/home/qinghua_mao/work/rllm/rllm/datasets/rel-movielens1m/classification/movies/test.csv"

class GenreOutputParser(BaseOutputParser):
    """Parse the output of LLM to a genre list"""

    def parse(self, text: str):
        """Parse the output of LLM call."""
        genres = text.split('::')[-1]
        genre_list = [genre.strip() for genre in genres.split(',')]
        return genre_list

output_parser = GenreOutputParser()

# Load documents from persist directory of vectorstore.
model_name = "all-MiniLM-L6-v2.gguf2.f16.gguf"
gpt4all_kwargs = {'allow_download': 'True'}
embeddings = GPT4AllEmbeddings(
    model_name = model_name,
    gpt4all_kwargs = gpt4all_kwargs
)
vectorstore = Chroma(persist_directory="/home/qinghua_mao/work/rllm/chroma", collection_name="rag-chroma", embedding_function=embeddings)

# retrieve and generate using the relevant snippets of the blog
retriever = vectorstore.as_retriever()

##### 2. LLM prediction
movie_df = pd.read_csv(test_path)

pred_genre_list = []
if args.prompt == 'title':
    for index, row in tqdm(movie_df.iterrows(), total=len(movie_df), desc="Processing Movies"):
        pred, prompt_cost = self_rag(movie_name=row['Title'], prompt="title", retriever=retriever)
        total_cost = total_cost + prompt_cost
        pred_genre_list.append(pred)

        total_cost = total_cost + get_llm_chat_cost(','.join(pred), 'output')
else:
    for index, row in tqdm(movie_df.iterrows(), total=len(movie_df), desc="Processing Movies"):
        
        pred, prompt_cost = self_rag(prompt="all", retriever=retriever, {"Title": row['Title'], "Director": row['Director'], "Year": row['Year'], 
                 "Genre": row['Genre'], "Cast": row['Cast'], "Runtime": row['Runtime'], 
                 "Languages": row['Languages'], "Certificate": row['Certificate'], 
                 "Plot": row['Plot']})
        total_cost = total_cost + prompt_cost
        pred_genre_list.append(pred)

        total_cost = total_cost + get_llm_chat_cost(','.join(pred), 'output')

##### 3. Calculate macro f1 score
# Get all genres
movie_genres = movie_df["Genre"].str.split("|")
all_genres = list(set([genre for genres in movie_genres for genre in genres]))

mlb = MultiLabelBinarizer(classes=all_genres)
real_genres_matrix = mlb.fit_transform(movie_genres)
pred_genres_matrix = mlb.fit_transform(pred_genre_list)
macro_f1 = macro_f1_score(real_genres_matrix, pred_genres_matrix)
micro_f1 = micro_f1_score(real_genres_matrix, pred_genres_matrix)

##### End time
time_end = time.time()

print(f"macro_f1: {macro_f1}")
print(f"micro_f1: {micro_f1}")
print(f"Total time: {time_end - time_start}s")
print(f"Total USD$: {total_cost}")