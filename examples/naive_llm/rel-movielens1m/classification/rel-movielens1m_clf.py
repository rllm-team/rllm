# -*- coding: ascii -*-

# Naive llm pipeline for classification task in rel-movielens1M
# Paper: 
# Title only: macro_f1: 0.251, micro_f1: 0.387
# Full info: macro_f1: 0.892, micro_f1: 0.884
# Runtime: Title only: 2990s; Full info: 6757s (on a single 6G GPU)
# Cost: Title only: $0.2722; Full info: $0.5996
# Description: Give llm movie name and limited genres, then ask llm which genres the movie should belong to.
# Usage: python rel-movielens1m_clf.py --prompt title/all

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

from rllm.utils import macro_f1_score, micro_f1_score, get_llm_chat_cost

##### Parse argument
parser = argparse.ArgumentParser()
parser.add_argument('--prompt', choices=['title', 'all'], 
                    default='title', help='Choose prompt type.')
args = parser.parse_args()

##### Start time
time_start = time.time()

##### Global variables
total_cost = 0
test_path = "your/test_file/path"
llm_model_path = "your/llm/path"

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
prompt_title = """Q: Now I have a movie name: {movie_name}. What's the genres it may belong to? 
Note: 
1. Give the answer as following format:
movie_name:: genre_1, genre_2..., genre_n
2. The answer must only be chosen from followings:'Documentary', 'Adventure', 'Comedy', 'Horror', 'War', 'Sci-Fi', 'Drama', 'Mystery', 'Western', 'Action', "Children's", 'Musical', 'Thriller', 'Crime', 'Film-Noir', 'Romance', 'Animation', 'Fantasy'
3. Don't saying anything else.
A: 
"""

prompt_all = """Q: Now I have a movie description: The movie titled '{Title}' is directed by {Director} and was released in {Year}. The genre of this movie is {Genre}, with main cast including {Cast}. It has a runtime of {Runtime} and languages used including {Languages}, with a Certificate rating of {Certificate}. The plot summary is as follows: {Plot} What's the genres it may belong to? 
Note: 
1. Give the answer as following format:
movie_name:: genre_1, genre_2..., genre_n
2. The answer must only be chosen from followings:'Documentary', 'Adventure', 'Comedy', 'Horror', 'War', 'Sci-Fi', 'Drama', 'Mystery', 'Western', 'Action', "Children's", 'Musical', 'Thriller', 'Crime', 'Film-Noir', 'Romance', 'Animation', 'Fantasy'
3. Don't saying anything else.
A: """

prompt_title_template = PromptTemplate(
    input_variables=["movie_name"], template=prompt_title
)

prompt_all_template = PromptTemplate(
    input_variables=["Title", "Director", "Year", "Genre", "Cast", "Runtime", "Languages", "Certificate", "Plot"], 
    template=prompt_all
)
# Construct chain
if args.prompt == 'title':
    chain = prompt_title_template | llm | output_parser
else:
    chain = prompt_all_template | llm | output_parser

##### 2. LLM prediction
movie_df = pd.read_csv(test_path)

pred_genre_list = []
if args.prompt == 'title':
    for index, row in tqdm(movie_df.iterrows(), total=len(movie_df), desc="Processing Movies"):
        total_cost = total_cost + get_llm_chat_cost(prompt_title_template.invoke({"movie_name": row['Title']}).text, 'input')

        pred = chain.invoke({"movie_name": row['Title']})
        pred_genre_list.append(pred)

        total_cost = total_cost + get_llm_chat_cost(','.join(pred), 'output')
else:
    for index, row in tqdm(movie_df.iterrows(), total=len(movie_df), desc="Processing Movies"):
        total_cost = total_cost + \
            get_llm_chat_cost(prompt_all_template.invoke(
                {"Title": row['Title'], "Director": row['Director'], "Year": row['Year'], 
                 "Genre": row['Genre'], "Cast": row['Cast'], "Runtime": row['Runtime'], 
                 "Languages": row['Languages'], "Certificate": row['Certificate'], 
                 "Plot": row['Plot']}).text, 'input')
        
        pred = chain.invoke({"Title": row['Title'], "Director": row['Director'], "Year": row['Year'], 
                 "Genre": row['Genre'], "Cast": row['Cast'], "Runtime": row['Runtime'], 
                 "Languages": row['Languages'], "Certificate": row['Certificate'], 
                 "Plot": row['Plot']})
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