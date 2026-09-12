#!/usr/bin/env python3
"""Run target-repository pytest and stream a durable test_logs record."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import shlex
import subprocess
import sys


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Run pytest in an explicit rllm repository and save combined output "
            "under the repository's test_logs/ directory."
        )
    )
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument(
        "--log-name",
        help="Log filename under test_logs/; defaults to a UTC timestamp.",
    )
    parser.add_argument(
        "pytest_args",
        nargs=argparse.REMAINDER,
        help="Arguments passed to pytest; place them after --.",
    )
    ns = parser.parse_args()
    pytest_args = ns.pytest_args
    if pytest_args[:1] == ["--"]:
        pytest_args = pytest_args[1:]
    return ns, pytest_args


def main() -> int:
    args, pytest_args = parse_args()
    repo = args.repo_root.expanduser().resolve()
    if not (repo / "rllm").is_dir() or not (repo / "test").is_dir():
        print(f"error: expected rllm/ and test/ under {repo}", file=sys.stderr)
        return 2

    log_dir = repo / "test_logs"
    log_dir.mkdir(exist_ok=True)
    default_name = datetime.now(timezone.utc).strftime("pytest-%Y%m%dT%H%M%SZ.log")
    log_name = args.log_name or default_name
    if Path(log_name).name != log_name:
        print("error: --log-name must be a filename, not a path", file=sys.stderr)
        return 2
    log_path = log_dir / log_name

    command = [sys.executable, "-m", "pytest", *pytest_args]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(repo), env["PYTHONPATH"]] if env.get("PYTHONPATH") else [str(repo)]
    )
    header = (
        f"utc_started={datetime.now(timezone.utc).isoformat()}\n"
        f"repo_root={repo}\n"
        f"python={sys.executable}\n"
        f"command={shlex.join(command)}\n"
        f"PYTHONPATH={env['PYTHONPATH']}\n\n"
    )

    try:
        with log_path.open("x", encoding="utf-8") as log:
            log.write(header)
            log.flush()
            print(header, end="")
            process = subprocess.Popen(
                command,
                cwd=repo,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="")
                log.write(line)
            returncode = process.wait()
            footer = (
                f"\nutc_finished={datetime.now(timezone.utc).isoformat()}\n"
                f"exit_code={returncode}\n"
            )
            print(footer, end="")
            log.write(footer)
    except FileExistsError:
        print(f"error: refusing to overwrite existing log: {log_path}", file=sys.stderr)
        return 2

    print(f"log_file={log_path}")
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
