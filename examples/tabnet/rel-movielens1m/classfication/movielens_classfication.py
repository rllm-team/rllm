# Tabnet for classification task in rel-movielens1M
# Paper: TabNet: Attentive Interpretable Tabular Learning  https://doi.org/10.1609/aaai.v35i8.16826
# arxiv : https://arxiv.org/abs/1908.07442
# macro_f1:0.062 , micro_f1: 0.130
# Runtime: 5.36s on single CPU (AMD Ryzen 5 5600U with Radeon Graphics 2.3Ghz)
# Cost: N/A
# Description: Simply apply TabNet to movielens.
import sys
sys.path.append("../../src")
from tab_model import TabNetMultiTaskClassifier

import time
import numpy as np
import torch
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score


# start time
time_start = time.time()

# 1.Load and process data
users = pd.read_csv(
    '../../../../rllm/datasets/rel-movielens1m/classification/users.csv')
train_movies = pd.read_csv(
    '../../../../rllm/datasets/rel-movielens1m/classification/movies/train.csv')
validation_movies = pd.read_csv(
    '../../../../rllm/datasets/rel-movielens1m/classification/movies/validation.csv')
test_movies = pd.read_csv(
    '../../../../rllm/datasets/rel-movielens1m/classification/movies/test.csv')
ratings = pd.read_csv(
    '../../../../rllm/datasets/rel-movielens1m/classification/ratings.csv')

train_genres = train_movies['Genre'].str.get_dummies(sep='|')
validation_genres = validation_movies['Genre'].str.get_dummies(sep='|')
test_genres = test_movies['Genre'].str.get_dummies(sep='|')

target = 'Genre'

# label encode categorical features and fill empty cells


def process_data(data):
    nunique = data.nunique()
    types = data.dtypes

    categorical_columns = []
    categorical_dims = {}
    for col in data.columns:
        if types[col] == 'object' or nunique[col] < 200:
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


train, categorical_columns, categorical_dims = process_data(train_movies)
valid, categorical_columns, categorical_dims = process_data(validation_movies)
test, categorical_columns, categorical_dims = process_data(test_movies)

# define categorical features for categorical embeddings
unused_feat = ['Url', 'MovielensID']

features = [col for col in train.columns if col not in unused_feat+[target]]

cat_idxs = [i for i, f in enumerate(features) if f in categorical_columns]

cat_dims = [categorical_dims[f]
            for i, f in enumerate(features) if f in categorical_columns]

# 2.Build Network

# Network parameters
clf = TabNetMultiTaskClassifier(cat_idxs=cat_idxs,
                                cat_dims=cat_dims,
                                cat_emb_dim=1,
                                optimizer_fn=torch.optim.Adam,
                                optimizer_params=dict(lr=2e-2),
                                scheduler_params={"step_size": 50,
                                                  "gamma": 0.9},
                                scheduler_fn=torch.optim.lr_scheduler.StepLR,
                                mask_type='entmax'  # "sparsemax"
                                )

# 3.Training

X_train = train[features].values
y_train = train_genres.values

X_valid = valid[features].values
y_valid = validation_genres.values

X_test = test[features].values
y_test = test_genres.values

clf.fit(
    X_train=X_train, y_train=y_train,
    eval_set=[(X_train, y_train), (X_valid, y_valid)],
    eval_name=['train', 'valid'],
    max_epochs=200, patience=20,
    batch_size=1024, virtual_batch_size=128,
    num_workers=0,
    drop_last=False,
    loss_fn=[torch.nn.functional.cross_entropy] *
    8  # Optional, just an example of list usage
)

# 4.Predictions
preds = clf.predict_proba(X_test)

# 5.Calculate f1 score
predict_classes = clf.predict(X_test)
predict_classes = np.transpose(predict_classes)
predict_classes = predict_classes.astype(int)

macro_f1 = f1_score(y_test, predict_classes, average='macro')
micro_f1 = f1_score(y_test, predict_classes, average='micro')

# End time
time_end = time.time()
print(f"Macro F1 Scores: {macro_f1}")
print(f"Micro F1 Scores: {micro_f1}")
print(f"Total time: {time_end - time_start}s")
