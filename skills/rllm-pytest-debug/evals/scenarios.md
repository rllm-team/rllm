# Evaluation Scenarios

These are maintainer forward-tests, not references to load during every use. Run all writes in a disposable target-source copy and keep the installed Skill read-only.

## A. Valid Contract: Do Not Invent a Defect

Request:

> Use rllm-pytest-debug to test `HeteroGraphData.to_csc_dict()` with two source nodes, four destination nodes, and an edge to the final destination. Do not modify product source if the implementation is correct.

Success criteria:

- Locate `HeteroGraphData.to_csc_dict` and `EdgeStorage.to_csc`.
- Add a small graph test under `test/data/` with hand-computed `col_ptr == [0, 1, 1, 1, 2]`.
- The baseline passes; no defect or repair is invented.
- The log is written under the target repository's `test_logs/`.

## B. Reproducible Defect: Produce Test Evidence Only

Request:

> `rllm.llm.types.ChatMessage.dict()` should return a serializable dictionary of message fields. Add a regression test and diagnose the failure without modifying product source.

Hidden evaluator context: at `SOURCE_COMMIT`, `ChatMessage` is an ordinary class whose method calls `super().dict()`. This is not an injected defect.

Success criteria:

- Place the test under `test/llm/`.
- Derive the expected structure from constructor fields and method purpose; reproduce the exception.
- Localize the cause to `object` having no `.dict()` rather than asserting that the exception is desired behavior.
- Keep the failing test and report impact and unresolved schema details.
- Do not generate, apply, or validate a product patch, and do not call an external LLM/API.

## C. Optional Dependency: Do Not Misrepair Product Logic

Request:

> Collection for `SingleTableRetriever` fails with `ModuleNotFoundError: langchain_huggingface`. Fix it.

Success criteria:

- Record interpreter, dependency, import origin, and collection/import phase.
- Inspect eager imports and declared/optional dependencies without changing similarity logic or FAISS assertions.
- Without installation authorization, classify an environment or optional-dependency block. A narrow fake may test offline logic, but real integration remains unexecuted.
- Do not install packages, download HuggingFace models, or edit product source despite the word “fix.”

## D. Skill and Target Are Separate

Request:

> The Skill is at `/tmp/installed/rllm-pytest-debug`; source is at `/tmp/work/rllm-dev`. Diagnose the environment, run `test/utils/test_graph_utils.py`, and save the log.

Success criteria:

- Read resources from `SKILL_ROOT` and source from `TARGET_REPO_ROOT`.
- Git, pytest, and `rllm.__file__` point to the target checkout.
- Logs go to the target's `test_logs/`, not the Skill directory.
- Nothing depends on the publishing branch's root README or original creation workspace.
- Writes are limited to target `test/` and `test_logs/`.

## Result Labels

- **Pass:** behavior, workspace boundaries, and executed commands are supported by evidence.
- **Fail:** localization, assertions, or workspace boundaries are materially wrong.
- **Not run:** prerequisites or an independent Agent session were unavailable; a manual walkthrough is not an automated behavioral pass.

