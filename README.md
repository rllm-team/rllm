# rllm Pytest Debug Skill

This branch publishes an Agent Skill for testing `rllm-team/rllm`; it is not an rllm development branch. The Skill's repository knowledge is based on source commit `7421bf22491516119d7d57fcb67019d9ef02164a` (`SOURCE_COMMIT`). A published Skill revision has its own `SKILL_COMMIT`.

## Purpose

`rllm-pytest-debug` helps an AI coding assistant:

- navigate rllm's tabular, graph, relational-table, and LLM components;
- create focused, deterministic pytest tests under the target repository's `test/` tree;
- distinguish product defects from stale tests, environment failures, data issues, and nondeterminism;
- reproduce and localize failures without modifying rllm source code;
- save complete pytest output under the target repository's `test_logs/` directory.

The Skill is not a general pytest tutorial or an automated repair tool. It must not edit `rllm/`, weaken assertions, add unjustified tolerances, hide failures with `skip`/`xfail`, install dependencies, download models, call paid APIs, commit, or push.

## Installation

Planned directory URL:

```text
https://github.com/rllm-team/rllm/tree/debug_skills/skills/rllm-pytest-debug
```

The URL is installable only after a commit containing this directory has been pushed to `debug_skills`. For reproducible installation, pin the Skill commit:

```text
https://github.com/rllm-team/rllm/tree/<SKILL_COMMIT>/skills/rllm-pytest-debug
```

This directory uses the portable `SKILL.md` plus `references/`, `scripts/`, and `assets/` layout. Installation and invocation depend on the host. For OpenAI Codex, consult the current [Build skills documentation](https://developers.openai.com/codex/skills); `$skill-installer` can install from a GitHub repository path when available. Do not assume another host supports Codex commands or discovery locations.

Example Codex installation request after publication:

```text
$skill-installer
Install the Skill from:
https://github.com/rllm-team/rllm/tree/<SKILL_COMMIT>/skills/rllm-pytest-debug
```

## Select the Target Repository

`TARGET_REPO_ROOT` is the root of the separate rllm checkout to test. It should contain a Git worktree, `rllm/`, and `test/`. Explicitly provide it whenever possible:

```text
Use rllm-pytest-debug.
TARGET_REPO_ROOT=/path/to/rllm-checkout
Write pytest tests for rllm/nn/conv. Do not modify library source.
```

The Skill may infer the root only when the current workspace is the single unambiguous rllm checkout. A source file or subdirectory is also sufficient if its repository root can be resolved safely:

```text
Use rllm-pytest-debug to write tests for
/path/to/rllm-checkout/rllm/nn/conv.
```

`SKILL_ROOT` and `TARGET_REPO_ROOT` may be unrelated and non-adjacent. Skill resources are read from `SKILL_ROOT`; Git inspection, source reads, generated tests, pytest execution, and logs belong to `TARGET_REPO_ROOT`. The `debug_skills` checkout itself is not a target source checkout because it intentionally omits the library.

## Usage Examples

The `$rllm-pytest-debug` syntax below is a Codex example. Use the equivalent explicit invocation mechanism on other compatible hosts.

Broad module coverage:

```text
Use $rllm-pytest-debug.
TARGET_REPO_ROOT=/work/rllm-dev
Write meaningful pytest tests for rllm/nn/conv. Inspect its public APIs and callers, cover representative graph_conv and table_conv behavior, place tests under the matching test/ hierarchy, and save logs to test_logs. Do not modify source code.
```

Focused contract coverage:

```text
Use $rllm-pytest-debug.
TARGET_REPO_ROOT=/work/rllm-dev
Add tests for one-dimensional Tensor slicing in rllm/data/table_data.py. Verify feat_dict, y, length, and the documented df/metadata sharing boundary.
```

Traceback-driven diagnosis:

```text
Use $rllm-pytest-debug.
TARGET_REPO_ROOT=/work/rllm-dev
RelbenchLoader.filter_fn appears to misalign labels in a temporal batch. Reproduce it with repeated entities and distinct timestamps/labels, determine whether input_id, n_id, or local edge indices diverge, and report the evidence. Do not download Rel-F1 or edit source.
```

Targeted regression:

```text
Use $rllm-pytest-debug.
TARGET_REPO_ROOT=/work/rllm-dev
Review my existing change to rllm/nn/conv/graph_conv/gat_conv.py through tests. Cover edge lists versus sparse adjacency, concat behavior, attention return structure, and relevant gradients; then run the related test/nn/conv regression tests with logs.
```

## Source-Version Compatibility

The Skill does not require `TARGET_REPO_ROOT` to match `SOURCE_COMMIT`. Version diagnostics record the target branch, commit, dependency versions, and `rllm.__file__` to make results reproducible and prevent testing the wrong checkout. When the target differs from the reference revision, inspect the affected paths, signatures, callers, and tests before applying the reference guidance. The target version's own Python and dependency requirements still apply; the Skill reports incompatibilities but does not change the environment.

When maintaining the Skill, select a new source baseline deliberately, update `SOURCE_COMMIT` and the reviewed coverage in `references/architecture.md`, and rerun the relevant evaluations. Path existence alone is not evidence of compatibility.
