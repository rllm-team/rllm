# Naive FT-transformer for classification task in rel-movielens1M
# Paper: Yury Gorishniy and Ivan Rubachev and Valentin Khrulkov and Artem Babenko (2021).
# Revisiting Deep Learning Models for Tabular Data arXiv preprint arXiv:2106.11959
# Test f1_score micro: 0.3240, macro: 0.1457
# Runtime: 16.655s on a 12GB GPU (NVIDIA(R) Tesla(TM) M40)
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
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing
import torch.nn.functional as F
import torch.optim
from torch import Tensor
from tqdm.std import tqdm
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.metrics import f1_score
import time
warnings.resetwarnings()

from rtdl_revisiting_models import FTTransformer

start_time = time.time()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Set random seeds in all libraries.
utils.random.seed(0)

# Dataset
# >>> Dataset.
TaskType = Literal["regression", "binclass", "multiclass"]
task_type: TaskType = "multiclass"

# Load Dataset
train_df = pd.read_csv(
    '../../../../rllm/datasets/rel-movielens1m/classification/movies/train.csv')
test_df = pd.read_csv(
    '../../../../rllm/datasets/rel-movielens1m/classification/movies/test.csv')
validation_df = pd.read_csv(
    '../../../../rllm/datasets/rel-movielens1m/classification/movies/validation.csv')

mlb = MultiLabelBinarizer()
mlb.fit(train_df['Genre'].str.split('|').apply(lambda x: set(x)))
genres_encoded = mlb.transform(
    train_df['Genre'].str.split('|').apply(lambda x: set(x)))
test_genres_encoded = mlb.transform(
    test_df['Genre'].str.split('|').apply(lambda x: set(x)))
validation_genres_encoded = mlb.transform(
    validation_df['Genre'].str.split('|').apply(lambda x: set(x)))

n_classes = len(mlb.classes_)

# apply to all datasets
scaler = StandardScaler().fit(train_df[['Year']])
train_df[['Year']] = scaler.transform(train_df[['Year']])
test_df[['Year']] = scaler.transform(test_df[['Year']])
validation_df[['Year']] = scaler.transform(validation_df[['Year']])

n_cont_features = 1

cat_features = ['Director', 'Cast', 'Runtime', 'Languages']
cat_cardinalities = [train_df[feature].nunique() for feature in cat_features]

X_cat = (
    np.column_stack(
        [np.random.randint(0, c, (len(cat_cardinalities),))
         for c in cat_cardinalities]
    )
    if cat_cardinalities
    else None
)

data_numpy = {
    'train': {'x_cont': torch.tensor(train_df[['Year']].values, dtype=torch.float),
              'y': torch.tensor(genres_encoded, dtype=torch.float)},
    'val': {'x_cont': torch.tensor(validation_df[['Year']].values, dtype=torch.float),
            'y': torch.tensor(validation_genres_encoded, dtype=torch.float)},
    'test': {'x_cont': torch.tensor(test_df[['Year']].values, dtype=torch.float),
             'y': torch.tensor(test_genres_encoded, dtype=torch.float)}
}

train_idx = range(len(train_df))

# Preprocessing
# >>> Feature preprocessing.
# NOTE
# The choice between preprocessing strategies depends on a task and a model.

# (A) Simple preprocessing strategy.
preprocessing = sklearn.preprocessing.StandardScaler().fit(
    data_numpy['train']['x_cont']
)

# (B) Fancy preprocessing strategy. NOT use because we do not need noise of float.
# The noise is added to improve the output of QuantileTransformer in some cases.
# X_cont_train_numpy = data_numpy["train"]["x_cont"]
# noise = (
#     np.random.default_rng(0)
#     .normal(0.0, 1e-5, X_cont_train_numpy.shape)
#     .astype(X_cont_train_numpy.dtype)
# )
# preprocessing = sklearn.preprocessing.QuantileTransformer(
#     n_quantiles=max(min(len(train_idx) // 30, 1000), 10),
#     output_distribution="normal",
#     subsample=10**9,
# ).fit(X_cont_train_numpy + noise)
# del X_cont_train_numpy

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
    part: {k: torch.as_tensor(v, device=device).float()
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
def evaluate(part: str) -> dict:
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

    y_pred_bin = (y_pred >= 0.5).astype(int)

    f1_micro = f1_score(y_true, y_pred_bin, average='micro')
    f1_macro = f1_score(y_true, y_pred_bin, average='macro')
    f1_weighted = f1_score(y_true, y_pred_bin, average='weighted')

    # The higher -- the better.
    return {'f1_micro': f1_micro, 'f1_macro': f1_macro, 'f1_weighted': f1_weighted}


score = evaluate("test")
print("Test set results before training: f1_test= {:.4f} {:.4f}".format(
    score["f1_micro"], score["f1_macro"]))

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

    val_score = evaluate("val")["f1_weighted"]
    test_score = evaluate("test")["f1_weighted"]
    f1_micro = evaluate("test")["f1_micro"]
    f1_macro = evaluate("test")["f1_macro"]
    print(
        f"(val) {val_score:.4f} (test) {test_score:.4f} (f1_micro) {f1_micro:.4f} (f1_marco) {f1_macro:.4f} [time] {timer}")

    early_stopping.update(val_score)
    if early_stopping.should_stop():
        break

    if val_score > best["val"]:
        print("🌸 New best epoch! 🌸")
        best = {"val": val_score, "test": test_score,
                "f1_micro": f1_micro, "f1_marco": f1_macro, "epoch": epoch}
    print()

print("\n\nResult: f1_micro:{f1_micro}, f1_marco:{f1_marco}".format(
    f1_micro=best["f1_micro"], f1_marco=best["f1_marco"]))
print("Time: ", time.time() - start_time)
