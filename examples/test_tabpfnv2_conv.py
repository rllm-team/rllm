#!/usr/bin/env python
"""Test script for TabPFN v2 Attention variant.

This script tests the attention version of TabPFN v2 to ensure
all components work correctly.
"""

import torch
import sys

# Add project to path
sys.path.insert(0, r"e:\MyOwnDoc\Project\PyProject\rllm")

from rllm.nn.conv.table_conv import (
    AttentionFeatureTransformer,
    AttentionEncoderLayer,
    PerFeatureTransformerAttn,
)


def test_attention_encoder_layer():
    """Test AttentionEncoderLayer"""
    print("\n" + "=" * 60)
    print("Testing AttentionEncoderLayer")
    print("=" * 60)

    # Create layer
    layer = AttentionEncoderLayer(
        d_model=128,
        nhead=8,
        dim_feedforward=512,
        hidden_dim=512,
        activation="gelu",
        dropout=0.1,
        use_feature_processor=True,
    )

    # Create sample input
    # Shape: (batch, num_items, num_features, d_model)
    batch_size, num_items, num_features, d_model = 4, 50, 10, 128
    x = torch.randn(batch_size, num_items, num_features, d_model)

    print(f"Input shape: {x.shape}")

    # Forward pass
    try:
        out = layer(x, single_eval_pos=40)
        print(f"Output shape: {out.shape}")
        assert out.shape == x.shape, f"Shape mismatch: {out.shape} vs {x.shape}"
        print("✓ AttentionEncoderLayer test passed!")
        return True
    except Exception as e:
        print(f"✗ AttentionEncoderLayer test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_attention_feature_transformer():
    """Test AttentionFeatureTransformer"""
    print("\n" + "=" * 60)
    print("Testing AttentionFeatureTransformer")
    print("=" * 60)

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

    print(f"Model created with {model.nlayers} attention layers")

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

        print(f"Output keys: {output.keys()}")
        print(f"Output shape: {output['standard'].shape}")

        # Verify output shape
        expected_test_size = seq_len - 40
        assert output["standard"].shape == (
            batch_size,
            expected_test_size,
            1,
        ), f"Shape mismatch: {output['standard'].shape}"

        print("✓ AttentionFeatureTransformer test passed!")
        return True

    except Exception as e:
        print(f"✗ AttentionFeatureTransformer test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_backward_pass():
    """Test backward pass for training"""
    print("\n" + "=" * 60)
    print("Testing Backward Pass (Training)")
    print("=" * 60)

    # Create model
    model = AttentionFeatureTransformer(
        emsize=64,
        nlayers=2,
        nhead=4,
        hidden_dim=256,
        activation="gelu",
        dropout=0.1,
    )

    # Create sample data
    seq_len, batch_size, num_features = 30, 2, 5
    x = torch.randn(seq_len, batch_size, num_features)
    y = torch.randint(0, 2, (seq_len, batch_size)).float()

    # Forward pass
    try:
        output = model(x=x, y=y, single_eval_pos=20)
        loss = output["standard"].sum()

        # Backward pass
        loss.backward()

        # Check gradients
        has_gradients = False
        for param in model.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_gradients = True
                break

        if has_gradients:
            print("✓ Backward pass successful - gradients computed!")
            return True
        else:
            print("✗ No gradients found!")
            return False

    except Exception as e:
        print(f"✗ Backward pass test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_compatibility_names():
    """Test backward compatibility names"""
    print("\n" + "=" * 60)
    print("Testing Compatibility Names")
    print("=" * 60)

    try:
        # Test that legacy name works
        model = PerFeatureTransformerAttn(emsize=128, nlayers=2, nhead=8)

        # Check it's the same class
        assert isinstance(
            model, AttentionFeatureTransformer
        ), "PerFeatureTransformerAttn should be AttentionFeatureTransformer"

        print("✓ Compatibility names test passed!")
        return True

    except Exception as e:
        print(f"✗ Compatibility names test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("TabPFN v2 Attention Variant - Test Suite")
    print("=" * 80)

    results = []

    try:
        results.append(("AttentionEncoderLayer", test_attention_encoder_layer()))
        results.append(
            ("AttentionFeatureTransformer", test_attention_feature_transformer())
        )
        results.append(("Backward Pass", test_backward_pass()))
        results.append(("Compatibility Names", test_compatibility_names()))
    except Exception as e:
        print(f"\nFatal error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{name:<30} {status}")

    print(f"\nTotal: {passed}/{total} tests passed")
    print("=" * 80)

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
