---
name: rllm-pytest-debug
description: >
  Write and maintain pytest tests for rllm-team/rllm, reproduce failures, diagnose test and environment issues, and localize likely defects. Use for module-level test generation, boundary contracts, traceback investigation, stale-test analysis, and targeted regression. Only modify tests and test logs; never modify library source.
---

# rllm Pytest Debug

## Separate the Skill from the Target

- `SKILL_ROOT`: the directory containing this file. Read only this Skill's `references/`, `scripts/`, and `assets/` from here.
- `TARGET_REPO_ROOT`: the rllm Git worktree under test. Source inspection, Git diagnostics, test creation, pytest execution, and `test_logs/` belong here.

Ask for `TARGET_REPO_ROOT` explicitly whenever possible. If the user gives a source file or subdirectory, resolve its Git root. Infer the target from the current workspace only when it is the single unambiguous checkout and contains both `rllm/` and `test/`. Never assume the two roots are adjacent, that the Skill lives inside the source repository, or that the target uses `debug_skills`.

## Write Boundary

Only create or adjust tests under `TARGET_REPO_ROOT/test/` and logs under `TARGET_REPO_ROOT/test_logs/`. Treat `rllm/`, examples, datasets, and all other product code as read-only. Do not generate, apply, or implement source patches. A request phrased as a repair task still ends with reproduction, classification, localization, and an evidence report. Source repair requires a separate task outside this Skill.

## Read References on Demand

- For module ownership, symbols, test mapping, or symptom routing, read [references/repository-map.md](references/repository-map.md).
- For end-to-end table, graph, or relational execution paths, read [references/architecture.md](references/architecture.md).
- For shapes, dtypes, indexing, mutation, missing values, time, or return contracts, read [references/contracts.md](references/contracts.md).
- For environment checks, test placement, commands, logging, fixtures, and mocks, read [references/testing.md](references/testing.md).

The references describe `SOURCE_COMMIT` and are navigation aids, not authority over another revision. For a different target commit, verify the affected paths, signatures, callers, and current tests first.

## Choose the Entry Mode

- **Discovery:** the user names a module or behavior but no failure. Inspect its public interfaces, callers, and implementation categories; then design representative normal, boundary, and failure tests.
- **Failure diagnosis:** the user provides a traceback, failing test, or unexpected behavior. Reproduce on unchanged product source, classify the failure, and preserve a minimal test that exposes it when the expected behavior has independent support.

Classify outcomes as product defect, stale or incorrect test, environment/dependency issue, data issue, nondeterminism, or unresolved requirement. A common implementation choice or an open design question is not automatically a defect. Write a failing assertion only when documentation, signatures, established callers, mathematics, data integrity, or an explicit user requirement supplies an independent expectation.

## Workflow

1. Record the target root, branch, HEAD, existing changes, Python/pytest/dependency versions, `rllm.__file__`, and the initial failure. Do not overwrite user work, install packages, download resources, commit, or push.
2. Read the implementation, direct callers, comparable components, and existing tests. Existing tests may be stale or weak; use them as history and coverage evidence, not as the quality standard.
3. Place new tests under `test/`, preferably mirroring the `rllm/` hierarchy. For a directory-level request, group tests by actual submodule rather than generating one superficial test per file.
4. Use small, deterministic, independently checkable inputs. Run the core assertion against unchanged product source. Do not claim a problem is confirmed if it cannot be reproduced.
5. Localize the failure to the smallest data-preparation, indexing, state, or module boundary. Mock external I/O, model downloads, and paid services only; do not mock the core logic under test.
6. Correct a stale test only after explaining the stronger contract evidence. Never weaken a valid assertion, inflate tolerances without evidence, swallow exceptions, or add convenience `skip`/`xfail` markers.
7. Run the new test, then the relevant file, module, and caller regressions. Preserve device-specific reproduction conditions; CPU success does not resolve a GPU failure.
8. Save every pytest run under `TARGET_REPO_ROOT/test_logs/`. Use `scripts/run_pytest_logged.py` or an equivalent command that preserves pytest's exit code.
9. Confirm that no product-source diff was introduced.

## Test Quality

- Assert contract-relevant values, ordering, alignment, masks, indices, state, errors, and return structure—not shape alone.
- Derive expected values independently of the implementation under test.
- Use `tmp_path` for files and caches. Fast tests should not use networks, model downloads, or external APIs.
- For neural components, check finite values, dtype/device consistency, relevant gradients, and train/eval behavior only where the contract requires them. Do not require every parameter to have a nonzero gradient or use short-training accuracy as a general unit-test oracle.
- Control random-state leakage and use justified numerical tolerances. Keep GPU-only issues on GPU.

## Completion Report

Report the symptom, expectation and evidence, classification, likely source location and impact, test/log changes, exact commands, log paths, results, pre-existing failures, skipped or unexecuted checks, and why. State explicitly that product source was not modified. Distinguish passed, failed, skipped, not run, and static-only findings.
