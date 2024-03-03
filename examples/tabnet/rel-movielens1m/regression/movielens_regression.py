# Tabnet for regression task in rel-movielens1M
# Paper: TabNet: Attentive Interpretable Tabular Learning  https://doi.org/10.1609/aaai.v35i8.16826
# arxiv : https://arxiv.org/abs/1908.07442
# mse:1.254
# Runtime: 30min20s on single CPU (Inter(R) Core(TM) i7-11800H @ 2.30Ghz)
# Cost: N/A
# Description: Simply apply TabNet to movielens.

import os
import sys
sys.path.append("../../src")
from tab_model import TabNetRegressor

import time
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import torch


np.random.seed(0)

# start time
time_start = time.time()

# 1.Load and process data
train = pd.read_csv(
    '../../../../rllm/datasets/rel-movielens1m/regression/ratings/train.csv')
validation = pd.read_csv(
    '../../../../rllm/datasets/rel-movielens1m//regression/ratings/validation.csv')
test = pd.read_csv(
    '../../../../rllm/datasets/rel-movielens1m/regression/ratings/test.csv')
users = pd.read_csv(
    '../../../../rllm/datasets/rel-movielens1m/regression/users.csv')
movies = pd.read_csv(
    '../../../../rllm/datasets/rel-movielens1m//regression/movies.csv')

# merge data through userID and MovieID


def merge_data(data):
    data_movies = pd.merge(
        data, movies, left_on='MovieID', right_on='MovielensID')

    data_movies_users = pd.merge(data_movies, users, on='UserID')

    return data_movies_users


train_merge = merge_data(train)
valid_merge = merge_data(validation)
test_merge = merge_data(test)

train_ratings = train_merge['Rating']
validation_ratings = valid_merge['Rating']
test_ratings = test_merge['Rating']

target = 'Rating'

# label encode categorical features and fill empty cells


def process_data(data):
    nunique = data.nunique()
    types = data.dtypes

    categorical_columns = []
    categorical_dims = {}
    for col in data.columns:
        if types[col] == 'object':
            print(col, data[col].nunique())
            data[col] = data[col].astype(str)
            l_enc = LabelEncoder()
            data[col] = data[col].fillna("VV_likely")
            data[col] = l_enc.fit_transform(data[col].values)
            categorical_columns.append(col)
            categorical_dims[col] = len(l_enc.classes_)
        else:
            data.fillna(data[col].mean(), inplace=True)
    return data, categorical_columns, categorical_dims


valid, categorical_columns, categorical_dims = process_data(valid_merge)
test, categorical_columns, categorical_dims = process_data(test_merge)
train, categorical_columns, categorical_dims = process_data(train_merge)

for col in categorical_columns:
    print(
        f"{col}: {train_merge[col].nunique()}, {valid_merge[col].nunique()}, {test_merge[col].nunique()}")

# define categorical features for categorical embeddings
unused_feat = ['Url', 'MovieID', 'Timestamp', 'Plot',
               'Title', 'Genre', 'Director', 'Cast', 'Language']
features = [col for col in train.columns if col not in unused_feat+[target]]

cat_idxs = [i for i, f in enumerate(features) if f in categorical_columns]

cat_dims = [categorical_dims[f]
            for i, f in enumerate(features) if f in categorical_columns]

# define embedding sizes
cat_emb_dim = [3, 4, 1, 1, 10]

# 2.Build Network

# Network parameters
clf = TabNetRegressor(
    cat_dims=cat_dims, cat_emb_dim=cat_emb_dim, cat_idxs=cat_idxs)

# 3.Training

X_train = train[features].values
y_train = train_ratings.values.reshape(-1, 1)

X_valid = valid[features].values
y_valid = validation_ratings.values.reshape(-1, 1)

X_test = test[features].values
y_test = test_ratings.values.reshape(-1, 1)


clf.fit(
    X_train=X_train, y_train=y_train,
    eval_set=[(X_train, y_train), (X_valid, y_valid)],
    eval_name=['train', 'valid'],
    eval_metric=['rmse'],
    max_epochs=2000,
    patience=50,
    batch_size=1024, virtual_batch_size=128,
    num_workers=0,
    drop_last=False
)

# 4.Predictions
preds = clf.predict(X_test)

# 5. Calculate MSE
test_mse = mean_squared_error(y_pred=preds, y_true=y_test)

# End time
time_end = time.time()
print(f"MSE : {test_mse}")
print(f"Total time: {time_end - time_start}s")
