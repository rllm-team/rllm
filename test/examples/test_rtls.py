# This file contains unit tests for relational table learning examples in the
# rLLM framework.
# The tests ensure that the example scripts run successfully and produce the
# expected output.

# The following example script is tested:
# 1. bridge.py: Tests the BRIDGE model on the TML1M, TLF2K and TACM12K datasets.

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
    script = os.path.join(EXAMPLE_ROOT, "bridge.py")
    out = subprocess.run(["python", str(script), "--dataset", "tml1m"], capture_output=True)
    assert (
        out.returncode == 0
    ), f"stdout: {out.stdout.decode('utf-8')}\nstderr: {out.stderr.decode('utf-8')}"
    stdout = out.stdout.decode("utf-8")
    assert float(stdout[-8:]) > 0.377


def test_bridge_tlf2k():
    script = os.path.join(EXAMPLE_ROOT, "bridge.py")
    out = subprocess.run(["python", str(script), "--dataset", "tlf2k"], capture_output=True)
    assert (
        out.returncode == 0
    ), f"stdout: {out.stdout.decode('utf-8')}\nstderr: {out.stderr.decode('utf-8')}"
    stdout = out.stdout.decode("utf-8")
    assert float(stdout[-8:]) > 0.451


def test_bridge_tacm12k():
    script = os.path.join(EXAMPLE_ROOT, "bridge.py")
    out = subprocess.run(["python", str(script), "--dataset", "tacm12k"], capture_output=True)
    assert (
        out.returncode == 0
    ), f"stdout: {out.stdout.decode('utf-8')}\nstderr: {out.stderr.decode('utf-8')}"
    stdout = out.stdout.decode("utf-8")
    assert float(stdout[-8:]) > 0.273
