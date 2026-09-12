# Contracts, Current Behavior, and Open Questions

Read this reference when an assertion depends on shape, dtype, indexing, mutation, missing values, time, or return structure. It describes `SOURCE_COMMIT` `7421bf22491516119d7d57fcb67019d9ef02164a`.

Use three evidence levels:

- **Supported contract:** signatures, documentation, callers, or tests agree strongly enough to protect directly.
- **Current behavior:** observable in the implementation but not necessarily a stable public promise.
- **Open design question:** evidence is incomplete or conflicting; a test author must not decide the API unilaterally.

An open question is not a bug list or a blocker. Accept internally consistent, common behavior unless it violates documentation, a signature, a mathematical invariant, data integrity, or an established caller. An observational test may record behavior without promoting it to a permanent API. Runtime errors, leakage, misalignment, and methods unable to perform their stated purpose are stronger defect candidates.

## TableData and Tensorization

### Supported contracts

- DataFrame rows are samples. After materialization, every `feat_dict` value and `y` align on first dimension `N`.
- With `concat=True`, numerical/categorical/binary groups are `[N, C]`; embedded text is `[N, C, D]`; tokenized text is `(input_ids, attention_mask)`, each `[N, C, L]`; timestamps are `[N, C, F]`.
- Numerical values become float32, categorical features become integer tensors, and the target is excluded from `feat_dict` and squeezed into `y`.
- Within a type, feature and metadata order follows `col_types.items()`. `TablePreEncoder` concatenates type groups in `feat_dict` iteration order.
- Indexing by `ColType` returns the full feature group. One-dimensional Tensor indexing creates a `TableData` whose features and labels match the selected rows.
- `_tensor_slice` shallow-shares `df` and metadata rather than recomputing statistics.
- Dataset and mask-based splits slice features and labels together; fractional splits must sum approximately to one.

Test with interleaved column types and distinct hand-computable values. Assert values, order, dtype, shape, labels, metadata, and intended identity boundaries.

### Current behavior

- `_unify_valid_pkey()` can mutate the caller's DataFrame through in-place index changes.
- `convert_text_coltypes` mutates the supplied `col_types` mapping.
- `shuffle()` reorders DataFrame rows, feature groups, and `y` in place, then resets the DataFrame index.
- Each categorical column uses independent sorted factorization; missing values map to `-1`. No persistent label map defines unseen-category behavior across separately materialized splits.
- Categorical targets may remain float32 until callers cast them for classification loss.
- `__len__`, `fkey_index`, and some transforms use caches whose invalidation after mutation is not uniformly specified.

When no reliable non-mutation contract exists, accept these observations rather than creating a failing test.

### Open questions

- Whether DataFrame and `col_types` mutation is public API.
- Whether `shuffle()` must preserve original primary-key semantics or only row alignment.
- Stable unseen-category handling across training and inference.
- `count_numerical_features()` is annotated/documented as a count but returns column names.
- Empty tables, target-only tables, single-class targets, and all-missing categorical columns lack consistent evidence.

Do not assume unsupported unseen-category or empty-input behavior merely because it would be desirable.

## Storage, Copying, and Devices

### Supported contracts

- `BaseStorage` supports `store.x` and `store["x"]`; underscore-prefixed attributes remain object state rather than ordinary mapping keys.
- `.apply()`, `.to()`, `.cpu()`, and `.cuda()` replace matching stored values in place and return the storage object. Recursive application covers tensors, sequences, and mappings.
- Shallow copies of `GraphData` and `HeteroGraphData` create new storage mappings while sharing values such as tensors; `clone()` additionally clones tensors.
- Heterogeneous node keys are strings and edge keys normalize to `(src, rel, dst)` triples.
- `HeteroGraphData.validate()` checks `[2, E]` edge indices, nonnegative IDs, and source/destination bounds. Isolated node types warn without making validation fail.

Test assignment and tensor mutation separately, inspect identity boundaries, check nested devices after `.to()`, and use asymmetric source/destination counts.

### Current behavior and open questions

- `NodeStorage.num_nodes` and `EdgeStorage.num_edges` infer sizes and return `-1` when inference fails.
- Node/edge attribute classification and total heterogeneous node count are cached; later mutation may leave stale classifications or totals.
- The guarantees of `validate()` around `-1` sizes, dtype, and sparse adjacency are incomplete.

## Edges, Sparse Layouts, and Message Passing

### Supported contracts

- Edge lists have shape `[2, E]`: row 0 is source and row 1 is destination.
- `_to_csc()` groups by destination: `col_ptr` indexes destination columns and `row` stores sources. Explicit `num_nodes` is the destination count; `HeteroGraphData.to_csc_dict()` passes the destination store size.
- `NeighborLoader` samples inbound neighbors, keeps seeds first in `n_id`, and expresses local edges in `n_id` coordinates.
- `MessagePassing.propagate()` emits destination-side representations; bipartite calls need destination features or explicit `dim_size` to preserve isolated destinations.
- Edge-list, dense, and sparse inputs are not universally interchangeable; test the layouts accepted by the concrete signature and implementation.

Use a bipartite graph with two sources and four destinations, include an edge to destination 3, and hand-check a length-five `col_ptr`. Include an isolated destination where output size matters.

### Current behavior

- Empty edge lists without explicit `num_nodes` reach `col.max()` in `_to_csc()`.
- `NeighborLoader` uses global Torch RNG. It stores `replace`, but the current one-layer sampler has no replacement branch.
- An empty hop may be represented as a one-dimensional empty long tensor rather than `[2, 0]` or a square sparse matrix.
- The legacy `utils.gcn_norm()` and `graph_transforms.GCNNorm` are different paths and must not be assumed equivalent from their names.

### Open questions

- The public promise for `replace=True`, empty-hop adjacency shape, and multi-hop `adjs` order.
- Whether directed GCN normalization follows in-degree or out-degree for each supported layout.
- Duplicate-edge reduction and propagation of non-unit edge attributes.

## Heterogeneous and RelBench Alignment

### Supported contracts

- RelBench graphs create FK-row-to-PK-entity edges and explicit reverse edge types.
- Task `input_id` is a split-row position; sampled `node` values are global entity IDs. They are not interchangeable.
- Temporal sampling is disjoint per seed and excludes future source nodes relative to seed time. The same global entity may appear in multiple seed subgraphs.
- Sampler `node` values are global IDs; `row` and `col` are local batch IDs. The loader slices stores by global IDs and writes local edges.
- Labels attach through `target[input_id]`, allowing the same entity to have different labels at different task times.
- `RelGNNModel` and `RDL` assume seed nodes occupy the target-node prefix and slice output by batch size or seed-time length.

Use repeated entities with distinct time/label pairs, a future edge, and an isolated table. Assert `input_id`, `n_id`, `y`, `seed_time`, batch ownership, and local edges—not just counts.

### Current behavior and open questions

- `RelbenchLoader` converts classification and regression targets to float for BCE/L1 examples.
- `get_loaders()` does not expose `use_pyg_lib`; the installed environment may choose a backend.
- `make_pkey_fkey_graph()` removes PK/FK entries from `col_types`, materializes tables, and writes caches. Featureless tables receive a numerical `__const__` column.
- The exact equal-time boundary, formal inductive/transductive visibility rules, and output-order equivalence between pyg-lib and the Python fallback are not centrally specified.

## Models, Gradients, and Modes

### Supported contracts

- Column encoders and `TablePreEncoder` emit `[B, C, H]`; `TableResNet` flattens columns and returns `[B, out_dim]`.
- `FTTransformerConv` returns `[B, C, H]` without CLS output and `[B, H]` with `use_cls=True`.
- GAT returns `[N_dst, heads * out_dim]` with concatenation and `[N_dst, out_dim]` otherwise; requested attention adds an edge/weight structure.
- Dropout, BatchNorm, and Transformer layers can differ between train and eval modes. Deterministic tests must set the mode and dropout deliberately.
- Gradient tests should use a scalar loss and require finite gradients only for parameters participating in the selected path.

Use small dimensions and either `dropout=0` or `eval()`. Assert finite output, dtype/device, contract-relevant structure, and independently justified values before gradients.

### Current behavior and open questions

- Some constructors reset child parameters more than once; an expected initialization distribution alone is weak defect evidence.
- `TableResNet` decoder comments describe a hidden dimension where the implementation emits `out_dim`.
- Accuracy thresholds after short training are slow, environment-sensitive integration smoke tests, not general unit contracts.

## LLM and Optional Dependencies

### Supported contracts

- `BaseLLM.chat()` returns `ChatResponse`; `complete()` returns `CompletionResponse`.
- `LangChainLLM` requires `langchain_core`; retrieval additionally uses HuggingFace/LangChain integration, FAISS, and model resources.
- Default text embedding can load a SentenceTransformer model; dataset construction may download and process data.
- Fast tests should inject fake tokenizers, embedders, or LLMs and mock adapter boundaries without accessing credentials or networks.

### Current behavior and open questions

- Concrete retrieval imports still load optional packages at module import time.
- At the baseline, `ChatMessage.dict()` calls `super().dict()` although `ChatMessage` inherits from `object`; it raises `AttributeError`. This is a reproducible defect candidate, while the intended schema and Pydantic compatibility remain open.
- Some LangChain chat models return message objects from `.invoke()`, whereas `LangChainLLM.complete()` passes that result directly to `CompletionResponse.text`.
- Supported LangChain/Transformers/SentenceTransformer matrices and external-service retry/timeout contracts are incomplete. Never probe them with paid calls.

## Resolving Contract Conflicts

1. Record what documentation, signatures, implementation, callers, and tests each imply.
2. If a test uses a missing argument while current signatures and examples agree, classify stale-test drift unless maintainers require backward compatibility.
3. If an implementation cannot perform its stated operation, write the smallest failing test and report a defect candidate plus unresolved schema details. Do not modify source.
4. If only current behavior is observable, label any test accordingly; do not promote it to a stable API.
5. Accept common implementation choices unless independent evidence shows an exception, invariant violation, broken caller, leakage, or misalignment.

