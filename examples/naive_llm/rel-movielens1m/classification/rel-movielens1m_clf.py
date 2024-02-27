# -*- coding: ascii -*-

# Naive llm pipeline for classification task in rel-movielens1M
# Paper: 
# macro_f1: 0.415, micro_f1: 0.494
# Runtime: 6097s (on a single 6G GPU)
# Cost: 0.2532 US Dollar
# Description: Give llm movie name and limited genres, then ask llm which genres the movie should belong to.

# Append rllm to search path
import sys
sys.path.append("../../../../")
import time

import pandas as pd

from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer

from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.schema import BaseOutputParser

from rllm.utils import macro_f1_score, micro_f1_score, get_llm_chat_cost

##### Start time
time_start = time.time()

##### Global variables
total_cost = 0
test_path = "rel-movielens1m/classification/movies/test.csv"
llm_model_path = "llama-2-7b-chat.Q4_K_M.gguf"

##### 1. Construct LLM chain
# Load model
llm = LlamaCpp(
    model_path=llm_model_path,
    streaming=False,
    n_gpu_layers=33,
    verbose=False,
    temperature=0.2,
    n_ctx=1024,
    stop=["\n"],
)

class GenreOutputParser(BaseOutputParser):
    """Parse the output of LLM to a genre list"""

    def parse(self, text: str):
        """Parse the output of LLM call."""
        genres = text.split('::')[-1]
        genre_list = [genre.strip() for genre in genres.split(',')]
        return genre_list

output_parser = GenreOutputParser()

# Construct prompt
prompt_zero_shot_classification = """Q: Now I have a movie name: {movie_name}. What's the genres it may belong to? 
Note: 
1. Give the answer as following format:
movie_name:: genre_1, genre_2..., genre_n
2. The answer must only be chosen from followings:'Documentary', 'Adventure', 'Comedy', 'Horror', 'War', 'Sci-Fi', 'Drama', 'Mystery', 'Western', 'Action', "Children's", 'Musical', 'Thriller', 'Crime', 'Film-Noir', 'Romance', 'Animation', 'Fantasy'
3. Don't saying anything else.
A: 
"""
prompt_template = PromptTemplate(
    input_variables=["movie_name"], template=prompt_zero_shot_classification
)

# Construct chain
chain = prompt_template | llm | output_parser

##### 2. LLM prediction
movie_df = pd.read_csv(test_path)
movie_names = movie_df["Title"]

pred_genre_list = []
for movie_name in tqdm(movie_names, total=len(movie_names), desc="Processing Movies"):
    total_cost = total_cost + get_llm_chat_cost(prompt_template.invoke({"movie_name": movie_name}).text, 'input')

    pred = chain.invoke({"movie_name": movie_name})
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