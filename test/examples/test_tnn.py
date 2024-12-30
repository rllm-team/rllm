# This file contains unit tests for various Table Neural Network (TNN) examples
# in the rLLM framework. The tests ensure that the example scripts run
# successfully and produce the expected output.

# The following example scripts are tested:
# 1. ft_transformer.py: Tests the FT-Transformer model.
# 2. tab_transformer.py: Tests the Tab-Transformer model.
# 3. tabnet.py: Tests the TabNet model.
# 4. excelformer.py: Tests the ExcelFormer model.
# 5. trompt.py: Tests the Trompt model.
# 6. saint.py: Tests the SAINT model.

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


def test_ft_transformer():
    script = os.path.join(EXAMPLE_ROOT, "ft_transformer.py")
    out = subprocess.run(["python", str(script)], capture_output=True)
    assert (
        out.returncode == 0
    ), f"stdout: {out.stdout.decode('utf-8')}\nstderr: {out.stderr.decode('utf-8')}"
    stdout = out.stdout.decode("utf-8")
    assert float(stdout[-8:]) > 0.80


def test_tab_transformer():
    script = os.path.join(EXAMPLE_ROOT, "tab_transformer.py")
    out = subprocess.run(["python", str(script)], capture_output=True)
    assert (
        out.returncode == 0
    ), f"stdout: {out.stdout.decode('utf-8')}\nstderr: {out.stderr.decode('utf-8')}"
    stdout = out.stdout.decode("utf-8")
    assert float(stdout[-8:]) > 0.80


def test_tabnet():
    script = os.path.join(EXAMPLE_ROOT, "tabnet.py")
    out = subprocess.run(["python", str(script)], capture_output=True)
    assert (
        out.returncode == 0
    ), f"stdout: {out.stdout.decode('utf-8')}\nstderr: {out.stderr.decode('utf-8')}"
    stdout = out.stdout.decode("utf-8")
    assert float(stdout[-8:]) > 0.74


def test_excel_former():
    script = os.path.join(EXAMPLE_ROOT, "excelformer.py")
    out = subprocess.run(["python", str(script)], capture_output=True)
    assert (
        out.returncode == 0
    ), f"stdout: {out.stdout.decode('utf-8')}\nstderr: {out.stderr.decode('utf-8')}"
    stdout = out.stdout.decode("utf-8")
    assert float(stdout[-8:]) > 0.80


def test_trompt():
    script = os.path.join(EXAMPLE_ROOT, "trompt.py")
    out = subprocess.run(["python", str(script)], capture_output=True)
    assert (
        out.returncode == 0
    ), f"stdout: {out.stdout.decode('utf-8')}\nstderr: {out.stderr.decode('utf-8')}"
    stdout = out.stdout.decode("utf-8")
    assert float(stdout[-8:]) > 0.80


def test_saint():
    script = os.path.join(EXAMPLE_ROOT, "saint.py")
    out = subprocess.run(["python", str(script)], capture_output=True)
    assert (
        out.returncode == 0
    ), f"stdout: {out.stdout.decode('utf-8')}\nstderr: {out.stderr.decode('utf-8')}"
    stdout = out.stdout.decode("utf-8")
    assert float(stdout[-8:]) > 0.89
