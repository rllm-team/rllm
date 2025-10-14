# This file contains unit tests for various Graph Neural Network (GNN) examples
# in the rLLM framework. The tests ensure that the example scripts run
# successfully and produce the expected output.

# The following example scripts are tested:
# 1. gcn.py: Tests the GCN model.
# 2. gat.py: Tests the GAT model.
# 3. rect.py: Tests the ReCT model.
# 4. ogc.py: Tests the OGC model.
# 5. han.py: Tests the HAN model.
# 6. hgt.py: Tests the HGT model.

# Each test function runs the corresponding example script and verifies the
# output to ensure it meets the expected criteria.

import os
import subprocess

EXAMPLE_ROOT = os.path.join(
    os.path.dirname(os.path.relpath(__file__)),
    "..",
    "..",
    "examples",
)


# Test homologous GNN
def test_gcn():
    script = os.path.join(EXAMPLE_ROOT, "gcn.py")
    out = subprocess.run(["python", str(script)], capture_output=True)
    assert (
        out.returncode == 0
    ), f"stdout: {out.stdout.decode('utf-8')}\nstderr: {out.stderr.decode('utf-8')}"
    stdout = out.stdout.decode("utf-8")
    assert float(stdout[-8:]) > 0.815


def test_gat():
    script = os.path.join(EXAMPLE_ROOT, "gat.py")
    out = subprocess.run(["python", str(script)], capture_output=True)
    assert (
        out.returncode == 0
    ), f"stdout: {out.stdout.decode('utf-8')}\nstderr: {out.stderr.decode('utf-8')}"
    stdout = out.stdout.decode("utf-8")
    assert float(stdout[-8:]) > 0.82


def test_rect():
    script = os.path.join(EXAMPLE_ROOT, "rect.py")
    out = subprocess.run(["python", str(script)], capture_output=True)
    assert (
        out.returncode == 0
    ), f"stdout: {out.stdout.decode('utf-8')}\nstderr: {out.stderr.decode('utf-8')}"
    stdout = out.stdout.decode("utf-8")
    assert float(stdout[-8:]) > 0.63


def test_ogc():
    script = os.path.join(EXAMPLE_ROOT, "ogc.py")
    out = subprocess.run(["python", str(script)], capture_output=True)
    assert (
        out.returncode == 0
    ), f"stdout: {out.stdout.decode('utf-8')}\nstderr: {out.stderr.decode('utf-8')}"
    stdout = out.stdout.decode("utf-8")
    assert float(stdout[-8:]) > 0.86


# Test heterogeneous GNN
def test_han():
    script = os.path.join(EXAMPLE_ROOT, "han.py")
    out = subprocess.run(["python", str(script)], capture_output=True)
    assert (
        out.returncode == 0
    ), f"stdout: {out.stdout.decode('utf-8')}\nstderr: {out.stderr.decode('utf-8')}"
    stdout = out.stdout.decode("utf-8")
    assert float(stdout[-8:]) > 0.56


def test_hgt():
    script = os.path.join(EXAMPLE_ROOT, "hgt.py")
    out = subprocess.run(["python", str(script)], capture_output=True)
    assert (
        out.returncode == 0
    ), f"stdout: {out.stdout.decode('utf-8')}\nstderr: {out.stderr.decode('utf-8')}"
    stdout = out.stdout.decode("utf-8")
    assert float(stdout[-8:]) > 0.56
