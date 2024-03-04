# -*- coding: ascii -*-

# Naive llm pipeline for regression task in rel-movielens1M
# Paper:
# MAE: 1.047
# Runtime: 97200s (on a single 6G GPU (Not applicable in practice))
# Cost: 7.6494 US Dollar
# Description: Use random 5 history rating to predict given user-movie rating by llm.
# Usage: python rel-movielens1m_reg.py

# Append rllm to search path
import sys
sys.path.append("../../../../")
import time

import pandas as pd
from tqdm import tqdm

from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from rllm.utils import mae, get_llm_chat_cost

##### Start time
time_start = time.time()

##### Global variables
total_cost = 0
train_path = "your/train_file/path"
movie_path = "your/movie_file/path"
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
    stop=["Q", "\n", " "],
)

# Output parser
output_parser = StrOutputParser()

# Construct prompt
prompt_zero_shot_regression = """Q: Given a user's past movie ratings in the format: Title, Genres, Rating
Ratings range from 1.0 to 5.0.

{history_ratings}

The candidate movie is {candidate}. What's the rating that the user will give? 
Give a single number as rating without saying anything else.
A: """

prompt_template = PromptTemplate(
    input_variables=["movie_name", "candidate"], template=prompt_zero_shot_regression)

# Construct chain
chain = prompt_template | llm | output_parser

##### 2. llm prediction
# Load files
test_data = pd.read_csv(test_path)
train_data = pd.read_csv(train_path)
movie_data = pd.read_csv(movie_path)


# Prediction
def FindMovieDetail(movie_data: pd.DataFrame, movie_id: int) -> str:
    # Find MID and Genres
    movie_info = movie_data[movie_data["MovielensID"] == movie_id]
    movie_name = movie_info["Title"].values[0]
    genres = movie_info["Genre"].values[0]

    return f"{movie_name}, {genres}"


predict_ratings = []
# Get each UID and MID
for index, row in tqdm(test_data.iterrows(), total=len(test_data), desc="Processing"):
    uid = row["UserID"]
    movie_id = row["MovieID"]

    # Find movie infomation
    movie_details = FindMovieDetail(movie_data, movie_id)

    # Find 5 random user history ratings
    user_ratings = train_data[train_data["UserID"] == uid].sample(n=5, random_state=42)
    history_movie_details_list = []

    # Get each MovieName and Genres
    for index, row in user_ratings.iterrows():
        movie_id = row["MovieID"]
        rating = row["Rating"]

        # Find history details
        history_movie_details = FindMovieDetail(movie_data, movie_id)
        history_movie_details = history_movie_details + f", {rating}"

        # Append history to list
        history_movie_details_list.append(history_movie_details)

    # use `\n` to concat
    history_movie_details_all = "\n".join(history_movie_details_list)

    total_cost = total_cost + get_llm_chat_cost(
        prompt_template.invoke(
            {"history_ratings": history_movie_details_all, "candidate": movie_details}
        ).text, 'input'
    )

    pred = chain.invoke(
        {"history_ratings": history_movie_details_all, "candidate": movie_details}
    )
    predict_ratings.append(float(pred))

    total_cost = total_cost + get_llm_chat_cost(pred, 'output')

##### 3. Calculate MAE
real_ratings = list(test_data["Rating"])
mae_loss = mae(real_ratings, predict_ratings)

##### End time
time_end = time.time()

print(mae_loss)
print(f"Total time: {time_end - time_start}s")
print(f"Total USD$: {total_cost}")
