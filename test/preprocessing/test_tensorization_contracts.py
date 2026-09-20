from __future__ import annotations

import pandas as pd
import torch

from rllm.preprocessing import df_to_tensor
from rllm.types import ColType


def test_df_to_tensor_keeps_declared_column_order_within_each_type():
    frame = pd.DataFrame(
        {
            "second_numeric": [2.0, 4.0],
            "category": ["beta", "alpha"],
            "first_numeric": [1.0, 3.0],
            "label": [1.0, 0.0],
        }
    )
    features, labels = df_to_tensor(
        frame,
        {
            "second_numeric": ColType.NUMERICAL,
            "category": ColType.CATEGORICAL,
            "first_numeric": ColType.NUMERICAL,
            "label": ColType.NUMERICAL,
        },
        target_col="label",
    )

    assert torch.equal(
        features[ColType.NUMERICAL], torch.tensor([[2.0, 1.0], [4.0, 3.0]])
    )
    assert features[ColType.CATEGORICAL].dtype == torch.int32
    assert torch.equal(features[ColType.CATEGORICAL], torch.tensor([[1], [0]], dtype=torch.int32))
    assert torch.equal(labels, torch.tensor([1.0, 0.0]))


def test_df_to_tensor_handles_missing_scalar_values_without_nonfinite_outputs():
    frame = pd.DataFrame(
        {
            "numeric": [1.0, None, 3.0],
            "category": ["known", None, "known"],
            "binary": ["yes", None, "no"],
        }
    )
    features, labels = df_to_tensor(
        frame,
        {
            "numeric": ColType.NUMERICAL,
            "category": ColType.CATEGORICAL,
            "binary": ColType.BINARY,
        },
        binary_true_values=["yes"],
    )

    assert labels is None
    for tensor in features.values():
        assert torch.isfinite(tensor).all()
    assert features[ColType.NUMERICAL].shape == (3, 1)
    assert features[ColType.CATEGORICAL].shape == (3, 1)
    assert features[ColType.BINARY].shape == (3, 1)
