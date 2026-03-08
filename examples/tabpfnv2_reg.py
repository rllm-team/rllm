import sys
import os.path as osp
import numpy as np

import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder


sys.path.append("./")
sys.path.append("../")
from rllm.types import ColType
from rllm.datasets import Titanic
from rllm.nn.models import TabPFNClassifier
from rllm.data_augment.ensemble_preprocessing import (
    EnsembleConfig,
    default_classifier_preprocessor_configs,
    fit_preprocessing,
)
from rllm.nn.models.tabpfn_v2.tabpfn.base import initialize_tabpfn_model

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

# Encode labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train.numpy())
n_classes = len(label_encoder.classes_)
print(f"Number of classes: {n_classes}, classes: {label_encoder.classes_}")

# Convert to numpy for preprocessing
X_train_np = x_train.numpy()
X_test_np = x_test.numpy()

# Generate ensemble configurations
n_estimators = 4
ensemble_configs = EnsembleConfig.generate_for_classification(
    n=n_estimators,
    subsample_size=min(10_000, len(X_train_np)),
    add_fingerprint_feature=True,
    feature_shift_decoder="shuffle",
    polynomial_features="no",
    max_index=len(X_train_np),
    preprocessor_configs=default_classifier_preprocessor_configs(),
    class_shift_method="shuffle",
    n_classes=n_classes,
    random_state=0,
)

# Load the model
model_path = osp.join(osp.dirname(osp.realpath(__file__)), "..", "models", "tabpfn_v2")
model, config = initialize_tabpfn_model(
    model_dir=model_path,
    model_type="clf",
    model_id=0,
    static_seed=0,
)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"Using device: {device}")

# Iterate through all preprocessing configurations and collect outputs
outputs = []
softmax_temperature = 0.9

print(f"Processing {n_estimators} ensemble configurations...")
for idx, (cfg, preprocessor, X_train_proc, y_train_proc, cat_ix_proc) in enumerate(
    fit_preprocessing(
        configs=ensemble_configs,
        X_train=X_train_np,
        y_train=y_train_encoded,
        random_state=0,
        cat_ix=cat_indx,
        n_workers=1,
        parallel_mode="block",
    )
):
    print(f"\nEnsemble {idx + 1}/{n_estimators}:")

    # Preprocess test data
    X_test_proc = preprocessor.transform(X_test_np).X

    # Convert to tensors
    X_train_tensor = torch.as_tensor(X_train_proc, dtype=torch.float32, device=device)
    X_test_tensor = torch.as_tensor(X_test_proc, dtype=torch.float32, device=device)
    y_train_tensor = torch.as_tensor(y_train_proc, dtype=torch.float32, device=device)

    # Concatenate train and test, add batch dimension
    X_full = torch.cat([X_train_tensor, X_test_tensor], dim=0).unsqueeze(1)

    # Direct forward pass
    with torch.inference_mode():
        output = model(
            None,  # style
            X_full,
            y_train_tensor,
            only_return_standard_out=True,
            categorical_inds=cat_ix_proc,
            single_eval_pos=len(y_train_tensor),
        )

    # Remove batch dimension
    output = output.squeeze(1)

    # Apply temperature scaling
    if softmax_temperature != 1:
        output = output[:, :n_classes].float() / softmax_temperature

    # Reverse class permutation if exists
    if cfg.class_permutation is not None:
        output = output[..., cfg.class_permutation]

    outputs.append(output)
    print(f"  Output shape: {output.shape}")

print(f"\nCollected {len(outputs)} outputs")

# Aggregate outputs (average before softmax, as in TabPFNClassifier)
average_before_softmax = False
if average_before_softmax:
    output_final = torch.stack(outputs).mean(dim=0)
    output_final = torch.nn.functional.softmax(output_final, dim=1)
else:
    # Softmax each output before averaging
    outputs = [torch.nn.functional.softmax(o, dim=1) for o in outputs]
    output_final = torch.stack(outputs).mean(dim=0)

# Convert to numpy
prediction_probabilities = output_final.float().cpu().numpy()

# Normalize to ensure probabilities sum to 1
prediction_probabilities = prediction_probabilities / prediction_probabilities.sum(
    axis=1, keepdims=True
)

# Evaluate
print("\n=== Results ===")
print("ROC AUC:", roc_auc_score(y_test, prediction_probabilities[:, 1]))

predictions = np.argmax(prediction_probabilities, axis=1)
print("Accuracy:", accuracy_score(y_test, predictions))
