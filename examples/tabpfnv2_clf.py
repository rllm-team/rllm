import sys
import os.path as osp

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


sys.path.append("./")
sys.path.append("../")
from rllm.types import ColType
from rllm.datasets import Titanic
from rllm.nn.models import TabPFNv2
from rllm.transforms.table_transforms import DefaultTableTransform

# Load dataset
path = osp.join(osp.dirname(osp.realpath(__file__)), "..", "data")
data = Titanic(cached_dir=path)[0]
data.shuffle()

transform = DefaultTableTransform()
data = transform(data)

# Split dataset, here the ratio of train-val-test is 80%-10%-10%
train_dataset, _, test_dataset = data.get_dataset(
    train_split=0.5, val_split=0.0, test_split=0.5
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

# Initialize and load the model
model_path = osp.join(osp.dirname(osp.realpath(__file__)), "..", "models", "tabpfn_v2")
model = TabPFNv2(
    model_dir=model_path,
    model_type="clf",
)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Fit the model (prepare ensemble configurations)
print(f"Preparing {model.n_estimators} ensemble configurations...")
model.fit(x_train, y_train, cat_ix=cat_indx, random_state=0)

# Evaluate
print("Making predictions...")
prediction_probabilities = model(x_test)
predictions = np.argmax(prediction_probabilities, axis=1)
print("Accuracy:", accuracy_score(y_test, predictions))
