#!/usr/bin/env python3
"""Read-only diagnostics for an explicitly selected rllm checkout."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


def run(repo: Path, *args: str) -> dict[str, Any]:
    completed = subprocess.run(
        list(args), cwd=repo, text=True, capture_output=True, check=False
    )
    return {
        "command": list(args),
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def module_info(name: str) -> dict[str, Any]:
    spec = importlib.util.find_spec(name)
    if spec is None:
        return {"available": False}
    result: dict[str, Any] = {"available": True, "origin": spec.origin}
    try:
        module = __import__(name)
    except Exception as exc:  # diagnostic: preserve the import failure
        result["import_error"] = f"{type(exc).__name__}: {exc}"
    else:
        result["version"] = getattr(module, "__version__", None)
        result["file"] = getattr(module, "__file__", None)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose a target rllm repository without modifying it."
    )
    parser.add_argument(
        "--repo-root",
        required=True,
        type=Path,
        help="Explicit path to the target rllm repository root.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = args.repo_root.expanduser().resolve()
    if not repo.is_dir():
        print(f"error: repository directory does not exist: {repo}", file=sys.stderr)
        return 2
    git_probe = run(repo, "git", "rev-parse", "--git-dir")
    if git_probe["returncode"] != 0:
        print(f"error: not a git worktree: {repo}", file=sys.stderr)
        return 2
    if not (repo / "rllm").is_dir() or not (repo / "test").is_dir():
        print(f"error: expected rllm/ and test/ under {repo}", file=sys.stderr)
        return 2

    old_cwd = Path.cwd()
    old_path = list(sys.path)
    try:
        os.chdir(repo)
        sys.path.insert(0, str(repo))
        modules = {
            name: module_info(name)
            for name in (
                "rllm",
                "pytest",
                "torch",
                "numpy",
                "pandas",
                "scipy",
                "sklearn",
                "transformers",
                "sentence_transformers",
                "pyarrow",
                "langchain",
            )
        }
    finally:
        os.chdir(old_cwd)
        sys.path[:] = old_path

    result = {
        "repo_root": str(repo),
        "python": {"executable": sys.executable, "version": sys.version},
        "git": {
            "root": run(repo, "git", "rev-parse", "--show-toplevel"),
            "branch": run(repo, "git", "branch", "--show-current"),
            "head": run(repo, "git", "rev-parse", "HEAD"),
            "status": run(repo, "git", "status", "--short", "--branch"),
            "remotes": run(repo, "git", "remote", "-v"),
        },
        "pytest_config_files": [
            name
            for name in ("pytest.ini", "pyproject.toml", "setup.cfg", "tox.ini")
            if (repo / name).is_file()
        ],
        "modules": modules,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))

    rllm_file = modules["rllm"].get("file")
    if not rllm_file:
        return 1
    try:
        Path(rllm_file).resolve().relative_to(repo)
    except ValueError:
        print("error: rllm imported outside the target repository", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
