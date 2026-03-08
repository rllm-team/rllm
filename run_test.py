#!/usr/bin/env python
"""Quick test runner for TabPFN v2 Attention model."""

import sys
import torch

# Add project to path
sys.path.insert(0, r"e:\MyOwnDoc\Project\PyProject\rllm")

from rllm.nn.conv.table_conv import AttentionFeatureTransformer

print("=" * 70)
print("Testing AttentionFeatureTransformer - Quick Test")
print("=" * 70)

# Create model
model = AttentionFeatureTransformer(
    emsize=128,
    nlayers=4,
    nhead=8,
    dim_feedforward=512,
    hidden_dim=512,
    activation="gelu",
    dropout=0.1,
    use_feature_processor=True,
    layer_dropout=True,
)

print(f"\n✓ Model created with {model.nlayers} attention layers")

# Create sample data
seq_len, batch_size, num_features = 50, 4, 10
x = torch.randn(seq_len, batch_size, num_features)
y = torch.randint(0, 2, (seq_len, batch_size)).float()

print(f"\nInput features shape: {x.shape}")
print(f"Target shape: {y.shape}")

# Forward pass
try:
    with torch.no_grad():
        output = model(
            x=x,
            y=y,
            single_eval_pos=40,
            only_return_standard_out=True,
        )

    print(f"\n✓ Output shape: {output['standard'].shape}")

    # Verify output shape
    expected_test_size = seq_len - 40
    expected_shape = (batch_size, expected_test_size, 1)

    if output["standard"].shape == expected_shape:
        print(f"✓ Shape matches expected: {expected_shape}")
        print("\n✓✓✓ AttentionFeatureTransformer test PASSED! ✓✓✓")
    else:
        print(
            f"✗ Shape mismatch. Expected {expected_shape}, got {output['standard'].shape}"
        )

except Exception as e:
    print(f"\n✗ Test failed with error: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

print("=" * 70)
