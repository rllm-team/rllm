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
    assert float(stdout[-9:]) > 0.80


def test_tab_transformer():
    script = os.path.join(EXAMPLE_ROOT, "tab_transformer.py")
    out = subprocess.run(["python", str(script)], capture_output=True)
    assert (
        out.returncode == 0
    ), f"stdout: {out.stdout.decode('utf-8')}\nstderr: {out.stderr.decode('utf-8')}"
    stdout = out.stdout.decode("utf-8")
    assert float(stdout[-9:]) > 0.80


def test_tabnet():
    script = os.path.join(EXAMPLE_ROOT, "tabnet.py")
    out = subprocess.run(["python", str(script)], capture_output=True)
    assert (
        out.returncode == 0
    ), f"stdout: {out.stdout.decode('utf-8')}\nstderr: {out.stderr.decode('utf-8')}"
    stdout = out.stdout.decode("utf-8")
    assert float(stdout[-9:]) > 0.80
