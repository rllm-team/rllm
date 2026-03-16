import sys
import os.path as osp

import numpy as np
import torch
from sklearn.metrics import mean_squared_error, r2_score


sys.path.append("./")
sys.path.append("../")
from rllm.types import ColType
from rllm.datasets import CaliforniaHousing
from rllm.nn.models import TabPFNv2
from rllm.transforms.table_transforms import DefaultTableTransform

# Load dataset
path = osp.join(osp.dirname(osp.realpath(__file__)), "..", "data")
data = CaliforniaHousing(cached_dir=path)[0]
data.shuffle()

transform = DefaultTableTransform()
data = transform(data)

# Split dataset, here the ratio of train-val-test is 50%-0%-50%
train_dataset, _, test_dataset = data.get_dataset(
    train_split=0.5, val_split=0.0, test_split=0.5
)

cat_indx = (
    list(range(len(data.metadata[ColType.CATEGORICAL])))
    if ColType.CATEGORICAL in data.metadata
    else []
)

feature_parts_train = []
feature_parts_test = []
if ColType.CATEGORICAL in train_dataset.feat_dict:
    feature_parts_train.append(train_dataset.feat_dict[ColType.CATEGORICAL])
    feature_parts_test.append(test_dataset.feat_dict[ColType.CATEGORICAL])
feature_parts_train.append(train_dataset.feat_dict[ColType.NUMERICAL])
feature_parts_test.append(test_dataset.feat_dict[ColType.NUMERICAL])

x_train = torch.cat(feature_parts_train, dim=1)
x_test = torch.cat(feature_parts_test, dim=1)
y_train = train_dataset.y
y_test = test_dataset.y

# Initialize and load the model
model_path = osp.join(osp.dirname(osp.realpath(__file__)), "..", "checkpoint", "tabpfn")
model = TabPFNv2(
    model_dir=model_path,
    model_type="reg",
)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Fit the model (prepare ensemble configurations)
print(f"Preparing {model.n_estimators} ensemble configurations...")
model.fit(x_train, y_train, cat_ix=cat_indx, random_state=0)

# Evaluate
print("Making predictions...")
predictions = model(x_test)
y_test_np = (
    y_test.detach().cpu().numpy()
    if isinstance(y_test, torch.Tensor)
    else np.asarray(y_test)
)
rmse = np.sqrt(mean_squared_error(y_test_np, predictions))
r2 = r2_score(y_test_np, predictions)
print(f"RMSE: {rmse:.4f}")
print(f"R2 Score: {r2:.4f}")
