"""Template to adapt under TARGET_REPO_ROOT/test/<matching source path>/.

This is not a validated rllm test. Replace the illustrative operation with the
target API and derive expected values independently from that implementation.
"""

import torch


def test_small_deterministic_contract(tmp_path):
    # Arrange: use a hand-checkable table or graph; isolate files in tmp_path.
    input_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    expected = torch.tensor([3.0, 7.0])

    # Act: call the target API here.
    actual = input_tensor.sum(dim=1)

    # Assert structure and an independently derived expected value.
    assert actual.shape == (2,)
    assert actual.dtype == expected.dtype
    assert actual.device == expected.device
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
