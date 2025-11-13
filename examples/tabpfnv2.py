import sys
import os.path as osp
import numpy as np

import torch
from sklearn.metrics import accuracy_score, roc_auc_score


sys.path.append("./")
sys.path.append("../")
from rllm.types import ColType
from rllm.datasets import Titanic
from rllm.nn.models import TabPFNClassifier

# Load dataset
path = osp.join(osp.dirname(osp.realpath(__file__)), "..", "data")
data = Titanic(cached_dir=path)[0]
data.shuffle()

# Split dataset
cat_indx = list(range(len(data.metadata[ColType.CATEGORICAL])))
x = torch.cat(
    [data.feat_dict[ColType.CATEGORICAL], data.feat_dict[ColType.NUMERICAL]], dim=1
)
y = data.y
x_train = x[: int(0.5 * len(x))]
x_test = x[int(0.5 * len(x)) :]
y_train = y[: int(0.5 * len(y))]
y_test = y[int(0.5 * len(y)) :]


# Initialize the classifier
model_path = osp.join(osp.dirname(osp.realpath(__file__)), "..", "models", "tabpfn_v2")
clf = TabPFNClassifier(model_path=model_path, categorical_features_indices=cat_indx)

# Fit the model
clf.fit(x_train, y_train)

# Predict probabilities
prediction_probabilities = clf.predict(x_test)
print("ROC AUC:", roc_auc_score(y_test, prediction_probabilities[:, 1]))

# Predict labels
predictions = np.argmax(prediction_probabilities, axis=1)
print("Accuracy:", accuracy_score(y_test, predictions))
