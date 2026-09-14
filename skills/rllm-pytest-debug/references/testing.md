# Pytest Environment and Execution

Read this reference when choosing test placement, commands, logging, fixtures, mocks, or when classifying collection and dependency failures. Repository facts and observed results refer to `SOURCE_COMMIT` `7421bf22491516119d7d57fcb67019d9ef02164a`.

## Baseline Repository State

- Tests are primarily under `test/data/`, `test/dataloader/`, `test/datasets/`, `test/examples/`, `test/nn/conv/`, `test/nn/encoder/`, and `test/utils/`.
- The baseline uses plain pytest functions and assertions. It has no shared `conftest.py`, custom markers, plugin configuration, or pytest settings in `pytest.ini`, `pyproject.toml`, `setup.cfg`, or `tox.ini`.
- Default discovery applies. `test/utils/seg_softmax.py` is not collected because it does not match `test_*.py`.
- `.github/workflows/test_examples.yml` installs pytest but executes examples directly; it does not invoke the pytest tree.
- `test/examples/` launches training scripts and asserts metric thresholds. These are slow, download-prone, random integration smoke tests rather than default unit tests.
- `Dataset.__init__` may download/process data when caches are absent. Dataset tests use real paths and `force_reload=True`.
- `test/data/test_table_data.py::test_text_embedding` can instantiate and download a real SentenceTransformer.
- Preprocessing, transforms, losses, LLMs, `RelbenchLoader`, and most relational models lack direct baseline pytest coverage.

Use existing tests as history and regression clues, not as a style or correctness authority.

## Observed Validation Environment

The analysis used the user-provided `lake` environment:

- Python 3.9.21 and pytest 8.4.2.
- Torch 2.8.0+cu128, torch-geometric 2.6.1, NumPy 2.0.2, pandas 2.3.3, SciPy 1.13.1, scikit-learn 1.6.1, Transformers 4.57.3, and OpenAI 2.30.0.
- `pytest-cov` was unavailable.
- `rllm.__file__` resolved inside the selected worktree.
- Declared bounds included Torch `>=2.1.0,<=2.3.0` and Transformers `<=4.30.0`; observed versions exceeded those bounds, so dynamic results are environment-qualified.

Collection command:

```bash
PYTHONPATH="$TARGET_REPO_ROOT" python -m pytest --collect-only -q "$TARGET_REPO_ROOT/test"
# exit 0; 45 tests collected
```

A representative offline subset produced `14 passed, 2 failed`. Both failures were stale-test drift:

- `test/dataloader/test_neighbor_loader.py` used `edge_index/num_samples/node_idx/num_nodes/return_oeid`; the current constructor used `data/num_neighbors/seeds`.
- `test/nn/conv/test_ft_transformer_convs.py` used `dim`; the current constructor used `conv_dim`.

Do not add product compatibility code merely to satisfy an old test unless backward compatibility is independently required.

## Establish the Target Baseline

Run from any directory, but pass both roots explicitly:

```bash
SKILL_ROOT=/path/to/installed/rllm-pytest-debug
TARGET_REPO_ROOT=/path/to/rllm-checkout
python "$SKILL_ROOT/scripts/diagnose_environment.py" \
  --repo-root "$TARGET_REPO_ROOT"
```

Record Git root, branch, HEAD, status, remotes, Python, pytest, dependency versions, pytest configuration, and `rllm.__file__`. If rllm imports outside `TARGET_REPO_ROOT`, correct the working directory or `PYTHONPATH` before drawing conclusions.

Activate a conda environment only when the user identifies it and it exists, for example `conda activate lake`. The Skill must not assume that environment, create environments, or install missing packages.

## Test Placement

Place tests under `test/`, preferably mirroring `rllm/`:

| Source | Preferred test location |
| --- | --- |
| `rllm/data/table_data.py` | `test/data/test_table_data.py` or a focused peer |
| `rllm/preprocessing/fillna.py` | `test/preprocessing/test_fillna.py` |
| `rllm/transforms/graph_transforms/gcn_norm.py` | `test/transforms/graph_transforms/test_gcn_norm.py` |
| `rllm/nn/conv/graph_conv/gat_conv.py` | `test/nn/conv/graph_conv/test_gat_conv.py` |
| `rllm/dataloader/relbench_loader.py` | `test/dataloader/test_relbench_loader.py` |
| `rllm/llm/types.py` | `test/llm/test_types.py` |

Do not create empty intermediate directories merely for symmetry. Inspect existing tests to avoid duplication, but do not preserve a flat or stale layout by default. Append only when an existing file is focused and well-structured.

A broad request such as “test `rllm/nn/conv`” is valid. Enumerate concrete modules and public exports, group them by shared abstraction and graph/table family, then select representative and boundary cases. Do not generate low-value import-or-shape tests merely to touch every file.

## Logged Test Runs

Collect first, run the minimal test, then expand regression scope:

```bash
python "$SKILL_ROOT/scripts/run_pytest_logged.py" \
  --repo-root "$TARGET_REPO_ROOT" \
  --log-name collect-target.log -- --collect-only -q test/path/test_file.py

python "$SKILL_ROOT/scripts/run_pytest_logged.py" \
  --repo-root "$TARGET_REPO_ROOT" \
  --log-name reproduce.log -- -q \
  test/path/test_file.py::test_exact_contract

python "$SKILL_ROOT/scripts/run_pytest_logged.py" \
  --repo-root "$TARGET_REPO_ROOT" \
  --log-name regression-module.log -- -q test/path/
```

The helper runs `current_interpreter -m pytest` from `TARGET_REPO_ROOT`, prepends the target to `PYTHONPATH`, streams combined output, refuses to overwrite an existing log, and returns pytest's exit status. Confirm the interpreter before use.

## Fast-Test Design

- **Tables:** in-memory DataFrames, explicit `col_types`, and distinct hand-computable values for order, dtype, NA, target, and metadata.
- **Graphs:** a few nodes, handwritten `[2, E]` edges, asymmetric source/destination counts, and one contract-relevant isolated/self-loop/direction boundary.
- **Samplers:** avoid random selection by requesting all neighbors, or save/restore RNG state; assert both global `n_id` and local edge IDs.
- **Models:** small dimensions, `dropout=0` or `eval()`, finite/device/shape checks plus independent value, order, state, or gradient evidence.
- **Files and caches:** use `tmp_path`; never write to shared `./data` or `./cached_dir` paths.

`assets/pytest_test_template.py` is a starting template, not a validated rllm test.

## Fixtures and Mocking

Useful fixtures may include a typed DataFrame, homogeneous graph, heterogeneous bipartite graph, fake RelBench task/dataset, and fake tokenizer/embedder/LLM. Keep fixtures local until multiple files genuinely share them.

Mock HTTP/downloads, external model loading, LangChain/API calls, sleeps, and unavailable GPU backends. Preserve adapter input/output assertions. Do not mock column dispatch, the encoder or convolution under test, global/local index mapping, or target attachment.

## Read-Only Product Boundary

The Skill may create or adjust `test/` and create `test_logs/`; product directories remain read-only. A failing test is a valid diagnostic outcome.

- Do not modify `rllm/`, examples, or product data processing, and do not provide or apply source patches.
- A request using the word “fix” still produces a reproduction, classification, root-cause evidence, impact scope, and unresolved contract questions—not a repair implementation.
- Correct a newly written invalid test when necessary. Before editing an existing test, explain the stronger evidence that makes it stale or incorrect.
- Compare product-source status before and after. Stop and report any new source change.

## Failure Classification

- Collection-time `ModuleNotFoundError`: distinguish required dependencies, eager optional imports, and missing declarations.
- Model/data download or DNS failure: mark external-resource blocking; use an injected fake for offline logic, while leaving the real integration unexecuted.
- CUDA OOM, missing GPU, or ABI mismatch: preserve device and version evidence; CPU success does not resolve a GPU issue.
- Dependency outside declared bounds: reproduce within bounds when feasible; otherwise qualify the result.
- Training example below an accuracy threshold: inspect seed, cache, epochs, versions, and device; do not lower the threshold merely to pass.

## Repository Improvements, Not Baseline Facts

- Separate deterministic unit tests from network/model, GPU, and long-training suites with explicit markers.
- Add an offline pytest subset to CI instead of only running examples.
- Rename undiscovered tests and add collection tests for optional imports.
- Ignore raw `test_logs/` locally and publish only concise failure summaries or CI artifacts.
