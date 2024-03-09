# Naive FT-transformer for regression task in rel-movielens1M
# Paper: Yury Gorishniy and Ivan Rubachev and Valentin Khrulkov and Artem Babenko (2021).
# Revisiting Deep Learning Models for Tabular Data arXiv preprint arXiv:2106.11959
# Test MSE Loss: 1.0725
# Runtime: 2772.838s on a 12GB GPU (NVIDIA(R) Tesla(TM) M40)
# Cost: N/A
# Description: Simply apply FT-transformer to movielens.
import sys
import os.path
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_file_dir)
sys.path.append(project_dir + "/../")

import math
import warnings
from typing import Dict, Literal
import torch
warnings.simplefilter("ignore")
import utils
import numpy as np
import pandas as pd
import scipy.special
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing
import time
import torch.nn.functional as F
import torch.optim
from torch import Tensor
from tqdm.std import tqdm

warnings.resetwarnings()

from rtdl_revisiting_models import FTTransformer

start_time = time.time()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Set random seeds in all libraries.
utils.random.seed(0)
# Dataset
# >>> Dataset.
TaskType = Literal["regression", "binclass", "multiclass"]

task_type: TaskType = "regression"
n_classes = None

# Load Dataset
train_df = pd.read_csv(
    '../../../../rllm/datasets/rel-movielens1m/regression/ratings/train.csv')
test_df = pd.read_csv(
    '../../../../rllm/datasets/rel-movielens1m/regression/ratings/test.csv')
validation_df = pd.read_csv(
    '../../../../rllm/datasets/rel-movielens1m/regression/ratings/validation.csv')

X_train = train_df[['UserID', 'MovieID']].values.astype(np.float32)
Y_train = train_df['Rating'].values.astype(np.float32)
n_cont_features = X_train.shape[1]

X_test = test_df[['UserID', 'MovieID']].values.astype(np.float32)
Y_test = test_df['Rating'].values.astype(np.float32)

X_val = validation_df[['UserID', 'MovieID']].values.astype(np.float32)
Y_val = validation_df['Rating'].values.astype(np.float32)

cat_cardinalities = [
    # NOTE: uncomment the two lines below to add two categorical features.
    # 4,  # Allowed values: [0, 1, 2, 3].
    # 7,  # Allowed values: [0, 1, 2, 3, 4, 5, 6].
]
X_cat = (
    np.column_stack(
        [np.random.randint(0, c, (len(X_train),)) for c in cat_cardinalities]
    )
    if cat_cardinalities
    else None
)

data_numpy = {
    "train": {"x_cont": X_train, "y": Y_train},
    "val": {"x_cont": X_val, "y": Y_val},
    "test": {"x_cont": X_test, "y": Y_test},
}

train_idx = range(len(train_df))

# Preprocessing
# >>> Feature preprocessing.
# NOTE
# The choice between preprocessing strategies depends on a task and a model.

# (A) Simple preprocessing strategy.
# preprocessing = sklearn.preprocessing.StandardScaler().fit(
#     data_numpy['train']['x_cont']
# )

# (B) Fancy preprocessing strategy.
# The noise is added to improve the output of QuantileTransformer in some cases.
X_cont_train_numpy = data_numpy["train"]["x_cont"]
noise = (
    np.random.default_rng(0)
    .normal(0.0, 1e-5, X_cont_train_numpy.shape)
    .astype(X_cont_train_numpy.dtype)
)
preprocessing = sklearn.preprocessing.QuantileTransformer(
    n_quantiles=max(min(len(train_idx) // 30, 1000), 10),
    output_distribution="normal",
    subsample=10**9,
).fit(X_cont_train_numpy + noise)
del X_cont_train_numpy

for part in data_numpy:
    data_numpy[part]["x_cont"] = preprocessing.transform(
        data_numpy[part]["x_cont"])

# >>> Label preprocessing.
if task_type == "regression":
    Y_mean = data_numpy["train"]["y"].mean().item()
    Y_std = data_numpy["train"]["y"].std().item()
    for part in data_numpy:
        data_numpy[part]["y"] = (data_numpy[part]["y"] - Y_mean) / Y_std

# >>> Convert data to tensors.
data = {
    part: {k: torch.as_tensor(v, device=device)
           for k, v in data_numpy[part].items()}
    for part in data_numpy
}

if task_type != "multiclass":
    # Required by F.binary_cross_entropy_with_logits
    for part in data:
        data[part]["y"] = data[part]["y"].float()


# Model
# The output size.
d_out = n_classes if task_type == "multiclass" else 1


model = FTTransformer(
    n_cont_features=n_cont_features,
    cat_cardinalities=None,
    d_out=d_out,
    **FTTransformer.get_default_kwargs(),
).to(device)
optimizer = model.make_default_optimizer()


# Training
def apply_model(batch: Dict[str, Tensor]) -> Tensor:
    if isinstance(model, FTTransformer):
        return model(batch["x_cont"], batch.get("x_cat")).squeeze(-1)

    else:
        raise RuntimeError(f"Unknown model type: {type(model)}")


loss_fn = (
    F.binary_cross_entropy_with_logits
    if task_type == "binclass"
    else F.cross_entropy
    if task_type == "multiclass"
    else F.mse_loss
)


@torch.no_grad()
def evaluate(part: str) -> float:
    model.eval()

    eval_batch_size = 8096
    y_pred = (
        torch.cat(
            [
                apply_model(batch)
                for batch in utils.iter_batches(data[part], eval_batch_size)
            ]
        )
        .cpu()
        .numpy()
    )
    y_true = data[part]["y"].cpu().numpy()

    if task_type == "binclass":
        y_pred = np.round(scipy.special.expit(y_pred))
        score = sklearn.metrics.accuracy_score(y_true, y_pred)
    elif task_type == "multiclass":
        y_pred = y_pred.argmax(1)
        score = sklearn.metrics.accuracy_score(y_true, y_pred)
    else:
        assert task_type == "regression"
        score = - sklearn.metrics.mean_squared_error(y_true, y_pred)
    return score  # The higher -- the better.


# For demonstration purposes (fast training and bad performance),
# one can set smaller values:
# n_epochs = 20
# patience = 2
n_epochs = 1_000_000_000
patience = 16

batch_size = 256
epoch_size = math.ceil(len(train_idx) / batch_size)
timer = utils.tools.Timer()
early_stopping = utils.tools.EarlyStopping(patience, mode="max")
best = {
    "val": -math.inf,
    "test": -math.inf,
    "epoch": -1,
}

print(f"Device: {device.type.upper()}")
print("-" * 88 + "\n")
timer.run()
for epoch in range(n_epochs):
    for batch in tqdm(
        utils.iter_batches(data["train"], batch_size, shuffle=True),
        desc=f"Epoch {epoch}",
        total=epoch_size,
    ):
        model.train()
        optimizer.zero_grad()
        loss = loss_fn(apply_model(batch), batch["y"])
        loss.backward()
        optimizer.step()

    val_score = evaluate("val")
    test_score = evaluate("test")
    print(f"(val) {-val_score:.4f} (test) {-test_score:.4f} [time] {timer}")

    early_stopping.update(val_score)
    if early_stopping.should_stop():
        break

    if val_score > best["val"]:
        print("🌸 New best epoch! 🌸")
        best = {"val": -val_score, "test": -test_score, "epoch": epoch}
    print()

print("\n\nResult:")
print("MSE: ", best["test"])
print("Time: ", time.time() - start_time)
