"""Test isolation for an explicitly selected rllm source checkout."""

from __future__ import annotations

import os
from pathlib import Path
import sys


TARGET_ENV = "RLLM_TARGET_REPO_ROOT"
target_value = os.environ.get(TARGET_ENV)
if not target_value:
    raise RuntimeError(
        f"Set {TARGET_ENV} to the rllm repository root before collecting tests."
    )

TARGET_REPO_ROOT = Path(target_value).expanduser().resolve()
TARGET_PACKAGE = TARGET_REPO_ROOT / "rllm"
if not (TARGET_PACKAGE / "__init__.py").is_file():
    raise RuntimeError(
        f"{TARGET_ENV}={TARGET_REPO_ROOT} does not contain rllm/__init__.py"
    )

sys.path.insert(0, str(TARGET_REPO_ROOT))

import rllm  # noqa: E402


resolved_import = Path(rllm.__file__).resolve()
if not resolved_import.is_relative_to(TARGET_PACKAGE):
    raise RuntimeError(
        "Imported rllm from the wrong location: "
        f"expected beneath {TARGET_PACKAGE}, got {resolved_import}"
    )


def pytest_report_header() -> str:
    return f"rllm source under test: {resolved_import}"
