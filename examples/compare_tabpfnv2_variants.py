#!/usr/bin/env python
"""Test script comparing TabPFN v2 Conv vs Attention variants.

This script demonstrates the differences between:
1. Convolutional version (tabpfnv2_conv.py)
2. PyTorch Multi-Head Attention version (tabpfnv2_attn.py)
"""

import torch
import sys
import time

sys.path.insert(0, r"e:\MyOwnDoc\Project\PyProject\rllm")

from rllm.nn.conv.table_conv import (
    ConvFeatureTransformer,
    AttentionFeatureTransformer,
)


def create_sample_data(seq_len=50, batch_size=4, num_features=10):
    """Create sample training data."""
    x = torch.randn(seq_len, batch_size, num_features)
    y = torch.randint(0, 2, (seq_len, batch_size)).float()
    return x, y


def benchmark_model(model, x, y, num_runs=5):
    """Benchmark a model's forward and backward pass."""
    times_forward = []
    times_backward = []

    for _ in range(num_runs):
        # Forward pass
        start = time.time()
        with torch.no_grad():
            output = model(x=x, y=y, single_eval_pos=40)
        times_forward.append(time.time() - start)

        # Backward pass
        start = time.time()
        loss = output["standard"].sum()
        loss.backward()
        times_backward.append(time.time() - start)

    return {
        "forward": sum(times_forward) / len(times_forward),
        "backward": sum(times_backward) / len(times_backward),
    }


def count_parameters(model):
    """Count total parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_comparison():
    """Compare Conv and Attention variants."""
    print("\n" + "=" * 80)
    print("TabPFN v2: Convolutional vs PyTorch Multi-Head Attention")
    print("=" * 80)

    # Create models with same configuration
    config = {
        "emsize": 128,
        "nlayers": 4,
        "activation": "gelu",
        "dropout": 0.1,
    }

    print("\nModel Configuration:")
    print(f"  Embedding Size: {config['emsize']}")
    print(f"  Number of Layers: {config['nlayers']}")
    print(f"  Activation: {config['activation']}")
    print(f"  Dropout: {config['dropout']}")

    # Create models
    print("\n" + "-" * 80)
    print("Creating models...")

    conv_model = ConvFeatureTransformer(**config)
    attn_model = AttentionFeatureTransformer(nhead=8, **config)  # 8 attention heads

    print("✓ Models created successfully")

    # Parameter count comparison
    print("\n" + "-" * 80)
    print("Parameter Count Comparison:")
    conv_params = count_parameters(conv_model)
    attn_params = count_parameters(attn_model)

    print(f"  Convolutional Model: {conv_params:>10,} parameters")
    print(f"  Attention Model:     {attn_params:>10,} parameters")
    print(
        f"  Difference:          {abs(attn_params - conv_params):>10,} ({(attn_params/conv_params-1)*100:+.1f}%)"
    )

    # Create sample data
    x, y = create_sample_data(seq_len=50, batch_size=4, num_features=10)

    # Forward pass test
    print("\n" + "-" * 80)
    print("Forward Pass Test:")

    try:
        with torch.no_grad():
            output_conv = conv_model(x=x, y=y, single_eval_pos=40)
            output_attn = attn_model(x=x, y=y, single_eval_pos=40)

        print(f"  Conv output shape:  {output_conv['standard'].shape}")
        print(f"  Attn output shape:  {output_attn['standard'].shape}")
        print("  ✓ Both models produce valid outputs")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

    # Backward pass test
    print("\n" + "-" * 80)
    print("Backward Pass Test:")

    try:
        # Conv model
        output_conv = conv_model(x=x, y=y, single_eval_pos=40)
        loss_conv = output_conv["standard"].sum()
        loss_conv.backward()

        # Attn model
        output_attn = attn_model(x=x, y=y, single_eval_pos=40)
        loss_attn = output_attn["standard"].sum()
        loss_attn.backward()

        print("  ✓ Both models support backpropagation")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

    # Benchmark test
    print("\n" + "-" * 80)
    print("Performance Benchmark (5 runs):")

    # Clear gradients
    conv_model.zero_grad()
    attn_model.zero_grad()

    print("\n  Convolutional Model:")
    times_conv = benchmark_model(conv_model, x, y)
    print(f"    Forward:  {times_conv['forward']*1000:.2f} ms")
    print(f"    Backward: {times_conv['backward']*1000:.2f} ms")
    print(
        f"    Total:    {(times_conv['forward'] + times_conv['backward'])*1000:.2f} ms"
    )

    print("\n  Attention Model:")
    times_attn = benchmark_model(attn_model, x, y)
    print(f"    Forward:  {times_attn['forward']*1000:.2f} ms")
    print(f"    Backward: {times_attn['backward']*1000:.2f} ms")
    print(
        f"    Total:    {(times_attn['forward'] + times_attn['backward'])*1000:.2f} ms"
    )

    # Analysis
    print("\n" + "-" * 80)
    print("Analysis:")

    speedup_forward = times_conv["forward"] / times_attn["forward"]
    speedup_backward = times_conv["backward"] / times_attn["backward"]
    speedup_total = (times_conv["forward"] + times_conv["backward"]) / (
        times_attn["forward"] + times_attn["backward"]
    )

    print(f"\n  Speedup (Conv vs Attn):")
    print(f"    Forward:  {speedup_forward:.2f}x")
    print(f"    Backward: {speedup_backward:.2f}x")
    print(f"    Total:    {speedup_total:.2f}x")

    # Summary
    print("\n" + "=" * 80)
    print("Summary:")
    print("-" * 80)
    print(
        """
  Convolutional Variant (tabpfnv2_conv.py):
    ✓ Faster computation
    ✓ Fewer parameters
    ✓ Simpler architecture
    ✗ May lose some feature interaction capacity
  
  Attention Variant (tabpfnv2_attn.py):
    ✓ Full feature interaction (multi-head attention)
    ✓ PyTorch native implementation (well-optimized)
    ✓ Better interpretability
    ✓ Slower but more expressive
    ✗ More parameters
  
  Use Conv variant when: Speed and efficiency are critical
  Use Attn variant when: Model expressiveness is more important
"""
    )
    print("=" * 80)

    return True


def usage_example():
    """Show usage examples for both variants."""
    print("\n" + "=" * 80)
    print("Usage Examples")
    print("=" * 80)

    print("\n1. Convolutional Variant (Faster):")
    print("-" * 80)
    print(
        """
from rllm.nn.conv.table_conv import ConvFeatureTransformer

model = ConvFeatureTransformer(
    emsize=128,
    nlayers=10,
    kernel_size=3,
    activation='gelu',
    dropout=0.1,
)

# Forward pass
output = model(x=features, y=labels, single_eval_pos=train_size)
predictions = output['standard']  # Shape: (batch, test_size, 1)
    """
    )

    print("\n2. Attention Variant (More Expressive):")
    print("-" * 80)
    print(
        """
from rllm.nn.conv.table_conv import AttentionFeatureTransformer

model = AttentionFeatureTransformer(
    emsize=128,
    nlayers=10,
    nhead=8,  # Number of attention heads
    activation='gelu',
    dropout=0.1,
)

# Forward pass
output = model(x=features, y=labels, single_eval_pos=train_size)
predictions = output['standard']  # Shape: (batch, test_size, 1)
    """
    )

    print("\n3. Interchangeable API:")
    print("-" * 80)
    print(
        """
# Both models have the same interface
models = [
    ConvFeatureTransformer(emsize=128, nlayers=10),
    AttentionFeatureTransformer(emsize=128, nlayers=10, nhead=8),
]

for model in models:
    output = model(x, y, single_eval_pos=40)
    # Works identically for both
    """
    )


if __name__ == "__main__":
    try:
        test_comparison()
        usage_example()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
