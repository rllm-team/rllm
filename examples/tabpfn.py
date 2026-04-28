#!/usr/bin/env python
# The TabPFN method from the
# "Accurate predictions on small data with a tabular foundation model" paper.
# https://www.nature.com/articles/s41586-024-08328-6

# Datasets      Titanic
# Metrics       Acc
# Rept.         -
# Ours          0.8483
# Time(s)       9.09

from __future__ import annotations

import argparse
import os.path as osp
import sys

import numpy as np
import torch
from sklearn.metrics import accuracy_score

sys.path.append("./")
sys.path.append("../")

from rllm.datasets import Adult, Titanic
from rllm.nn.models import TabPFN
from rllm.transforms.table_transforms import DefaultTableTransform
from rllm.types import ColType

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset", type=str, default="titanic", choices=["titanic", "adult"]
)
parser.add_argument("--n_estimators", type=int, default=8)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

# Set device
torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
path = osp.join(osp.dirname(osp.realpath(__file__)), "..", "data")
if args.dataset.lower() == "adult":
    data = Adult(cached_dir=path)[0]
    metric = "auc"
else:
    data = Titanic(cached_dir=path)[0]
    metric = "acc"

transform = DefaultTableTransform()
data = transform(data)
data.shuffle()

train_dataset, _, test_dataset = data.get_dataset(
    train_split=0.8,
    val_split=0.0,
    test_split=0.2,
)

cat_indx = list(range(len(data.metadata[ColType.CATEGORICAL])))
x_train = torch.cat(
    [
        train_dataset.feat_dict[ColType.CATEGORICAL],
        train_dataset.feat_dict[ColType.NUMERICAL],
    ],
    dim=1,
)
x_test = torch.cat(
    [
        test_dataset.feat_dict[ColType.CATEGORICAL],
        test_dataset.feat_dict[ColType.NUMERICAL],
    ],
    dim=1,
)
y_train = train_dataset.y
y_test = test_dataset.y

# Load model
model_path = osp.join(osp.dirname(osp.realpath(__file__)), "..", "checkpoint", "tabpfn")
model = TabPFN(
    model_dir=model_path,
    model_type="clf",
    model_id=0,
    static_seed=args.seed,
    n_estimators=args.n_estimators,
    metadata=data.metadata,
    enable_flash_attention=False,
    inference_batch_size=4096,
    inference_precision=torch.float32,
    strict_version_match=True,
)
model = model.to(device)

print(f"Preparing {model.n_estimators} ensemble configurations...")
model.fit(x_train, y_train, cat_ix=cat_indx, random_state=args.seed)

print("Predicting...")
proba = model.predict_proba(x_test)
pred = np.argmax(proba, axis=1)
print("Accuracy:", accuracy_score(y_test, pred))
