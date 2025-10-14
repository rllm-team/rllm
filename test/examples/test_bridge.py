# This file contains unit tests for the BRIDGE examples in the rLLM framework.
# The tests ensure that the example scripts run successfully and produce the
# expected output.

# The following example scripts are tested:
# 1. bridge_tml1m.py: Tests the BRIDGE model on the TML1M dataset.
# 2. bridge_tlf2k.py: Tests the BRIDGE model on the TLF2K dataset.
# 3. bridge_tacm12k.py: Tests the BRIDGE model on the TACM12K dataset.

# Each test function runs the corresponding example script and verifies the
# output to ensure it meets the expected criteria.

import os
import subprocess

EXAMPLE_ROOT = os.path.join(
    os.path.dirname(os.path.relpath(__file__)),
    "..",
    "..",
    "examples",
    "bridge",
)


def test_bridge_tml1m():
    script = os.path.join(EXAMPLE_ROOT, "bridge_tml1m.py")
    out = subprocess.run(["python", str(script)], capture_output=True)
    assert (
        out.returncode == 0
    ), f"stdout: {out.stdout.decode('utf-8')}\nstderr: {out.stderr.decode('utf-8')}"
    stdout = out.stdout.decode("utf-8")
    assert float(stdout[-8:]) > 0.377


def test_bridge_tlf2k():
    script = os.path.join(EXAMPLE_ROOT, "bridge_tlf2k.py")
    out = subprocess.run(["python", str(script)], capture_output=True)
    assert (
        out.returncode == 0
    ), f"stdout: {out.stdout.decode('utf-8')}\nstderr: {out.stderr.decode('utf-8')}"
    stdout = out.stdout.decode("utf-8")
    assert float(stdout[-8:]) > 0.451


def test_bridge_tacm12k():
    script = os.path.join(EXAMPLE_ROOT, "bridge_tacm12k.py")
    out = subprocess.run(["python", str(script)], capture_output=True)
    assert (
        out.returncode == 0
    ), f"stdout: {out.stdout.decode('utf-8')}\nstderr: {out.stderr.decode('utf-8')}"
    stdout = out.stdout.decode("utf-8")
    assert float(stdout[-8:]) > 0.273
