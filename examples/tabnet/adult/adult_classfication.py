# -*- coding: ascii -*-

# Tabnet for classification task in adult
# Paper: TabNet: Attentive Interpretable Tabular Learning  https://doi.org/10.1609/aaai.v35i8.16826
# arxiv : https://arxiv.org/abs/1908.07442
# macro_f1: 0.81, micro_f1:0.85
# Runtime: 336.08s on single CPU(AMD Ryzen 5 5600U with Radeon Graphics 2.3Ghz)
# Cost: N/A
# Description: Paper Reproduction.
import sys
sys.path.append("../src")
from tab_model import TabNetClassifier
import scipy
import time
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

np.random.seed(0)

# start time
time_start = time.time()

# 1.Load and process data

# train = pd.read_csv('dataset/adult.data')
train = pd.read_csv('dataset/adult.data')
target = ' <=50K'

# cutting dataset into train, validation, test set
if "Set" not in train.columns:
    train["Set"] = np.random.choice(
        ["train", "valid", "test"], p=[.8, .1, .1], size=(train.shape[0],))

train_indices = train[train.Set == "train"].index
valid_indices = train[train.Set == "valid"].index
test_indices = train[train.Set == "test"].index

# label encode categorical features and fill empty cells
nunique = train.nunique()
types = train.dtypes

categorical_columns = []
categorical_dims = {}
for col in train.columns:
    if types[col] == 'object' or nunique[col] < 200:
        print(col, train[col].nunique())
        l_enc = LabelEncoder()
        train[col] = train[col].fillna("VV_likely")
        train[col] = l_enc.fit_transform(train[col].values)
        categorical_columns.append(col)
        categorical_dims[col] = len(l_enc.classes_)
    else:
        train.fillna(train.loc[train_indices, col].mean(), inplace=True)

# define categorical features for categorical embeddings
unused_feat = ['Set']

features = [col for col in train.columns if col not in unused_feat+[target]]

cat_idxs = [i for i, f in enumerate(features) if f in categorical_columns]

cat_dims = [categorical_dims[f]
            for i, f in enumerate(features) if f in categorical_columns]

# 2.Build Network

# Network parameters
clf = TabNetClassifier(cat_idxs=cat_idxs,
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

X_train = train[features].values[train_indices]
y_train = train[target].values[train_indices]

X_valid = train[features].values[valid_indices]
y_valid = train[target].values[valid_indices]

X_test = train[features].values[test_indices]
y_test = train[target].values[test_indices]

max_epochs = 200
sparse_X_train = scipy.sparse.csr_matrix(
    X_train)  # Create a CSR matrix from X_train
sparse_X_valid = scipy.sparse.csr_matrix(
    X_valid)  # Create a CSR matrix from X_valid
clf.fit(
    X_train=sparse_X_train, y_train=y_train,
    eval_set=[(sparse_X_train, y_train), (sparse_X_valid, y_valid)],
    eval_name=['train', 'valid'],
    max_epochs=max_epochs, patience=20,
    batch_size=1024, virtual_batch_size=128,
    num_workers=0,
    weights=1,
    drop_last=False
)
# 4.Predictions
preds = clf.predict_proba(X_test)

# 5.Calculate f1 score

true_labels = y_test
pred_labels = preds.argmax(axis=1)

macro_f1 = f1_score(true_labels, pred_labels, average='macro')
micro_f1 = f1_score(true_labels, pred_labels, average='micro')

# End time
time_end = time.time()

print(f"Macro F1 Scores: {macro_f1}")
print(f"Micro F1 Scores: {micro_f1}")
print(f"Total time: {time_end - time_start}s")
