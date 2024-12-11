import os
import subprocess

EXAMPLE_ROOT = os.path.join(
    os.path.dirname(os.path.relpath(__file__)),
    "..",
    "..",
    "examples",
)


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


def test_han():
    script = os.path.join(EXAMPLE_ROOT, "han.py")
    out = subprocess.run(["python", str(script)], capture_output=True)
    assert (
        out.returncode == 0
    ), f"stdout: {out.stdout.decode('utf-8')}\nstderr: {out.stderr.decode('utf-8')}"
    stdout = out.stdout.decode("utf-8")
    assert float(stdout[-8:]) > 0.56


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
