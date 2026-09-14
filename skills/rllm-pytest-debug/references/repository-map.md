# Repository Map

Read this reference when ownership is unclear or when a symptom needs to be routed to likely implementation and test entry points. It is a navigation guide, not a root-cause diagnosis. Facts refer to `SOURCE_COMMIT` `7421bf22491516119d7d57fcb67019d9ef02164a`; verify paths and symbols against other revisions.

## Top-Level Layout

| Path | Responsibility | First inspection point |
| --- | --- | --- |
| `rllm/` | Library source; most subpackages expose symbols lazily | Concrete implementation and the subpackage `_LAZY_MODULES` table |
| `test/` | Pytest tree; 45 tests were collectable at the baseline | Matching `test/<package>/`, then relevant `test/examples/` |
| `examples/` | Executable training and inference callers | Model script, defaults, downloads, cache paths, and device selection |
| `docs/` | Sphinx API and tutorials | Record conflicts with source or tests; do not resolve them silently |
| `.github/workflows/` | CI behavior | Whether workflows actually invoke pytest |
| `requirements.txt`, `setup.py` | Declared environment | Version bounds and missing optional dependencies |

The baseline tests document history and coverage, but they are not a consistent or current quality template. Design new tests from supported behavior and independent expected values. Prefer a source-mirroring test hierarchy without copying stale parameters or weak assertions.

## Source Areas

### `rllm/data`

- `storage.py`: mapping/attribute access through `BaseStorage`; node/edge attribute inference; recursive device transfer; CSC conversion.
- `graph_data.py`: homogeneous `GraphData`; typed-node and `(src, rel, dst)` edge stores in `HeteroGraphData`; validation, collection, CSC, copy, and clone behavior.
- `table_data.py`: DataFrame, `col_types`, primary/foreign keys, `feat_dict`, metadata, labels, slicing, splits, and `TableDataset` integration.
- `view.py`: key/value/item views over storage.

Baseline tests: `test/data/test_graph_data.py`, `test/data/test_table_data.py`. Add focused tests for storage views, persistence, empty graphs, or copy semantics when relevant.

### `rllm/datasets`

- `dataset.py`: may call `download()` and `process()` during construction when caches are absent.
- Single-table datasets: `adult.py`, `titanic.py`, `bank_marketing.py`, `churn_modelling.py`, `jannis.py`.
- Graph datasets: `planetoid.py`, `dblp.py`, `imdb.py`, `tape.py`, `tagdataset.py`.
- Relational datasets: `sjtutables/`, `lakemlb/`, `relbench/`; `RelBenchDataset` manages tables, tasks, graph materialization, and statistics caches.

Baseline dataset tests use real cache paths and may download data. Do not treat them as offline unit tests.

### `rllm/preprocessing`

- `df_to_tensor.py:df_to_tensor`: dispatches by `ColType`, cleans values, fills missing data, encodes categorical/binary columns, processes text and timestamps, and groups tensors by type.
- `fillna.py`, `_type_convert.py`, `data_clean.py`: scalar cleaning and conversion.
- `text_tokenize.py`, `word_embedding.py`: injectable tokenizer/embedder boundaries whose defaults may require model resources.
- `timestamp.py:TimestampPreprocessor`: timestamp parsing and feature extraction.

The baseline has no dedicated `test/preprocessing/`. Prefer small in-memory DataFrames over dataset construction.

### `rllm/transforms`

- `table_transforms/`: NA handling and column transforms; several transforms replace `TableData.feat_dict`.
- `graph_transforms/`: composable node/edge/graph transforms; `GCNTransform` combines feature normalization and `GCNNorm`.
- `utils/`: common `BaseTransform`, class removal, and meta-path propagation.

The baseline has no dedicated `test/transforms/`. Test mutation, caching, self-loops, and normalization directly.

### `rllm/nn/encoder`

- `col_encoder/`: type-specific encoders sharing the `[B, C, D]` output convention.
- `table_pre_encoder.py:TablePreEncoder`: encodes each `feat_dict` group and concatenates columns in iteration order.
- `*_pre_encoder.py`: FT-Transformer, TabTransformer, ResNet, Trompt, and TransTab compositions.
- `heterotemporal_encoder.py`: aligns sampled-node time and batch ownership with seed time.
- `colate.py`, `column_aware_table_encoder.py`, `metartl_encoder.py`: column-aware and relational paths. `ColATE` exists in more than one implementation file; confirm the exported symbol.

Baseline tests: `test/nn/encoder/test_col_encoder.py`, `test/nn/encoder/test_table_encoder.py`.

### `rllm/nn/conv`

- `graph_conv/message_passing.py:MessagePassing`: edge-list/sparse propagation through `message -> aggregate -> update`, with an optional fused path.
- `graph_conv/`: GCN, GAT, GraphSAGE, LGC, HAN, HGT, graph Transformer, RelGNN, and aggregation implementations.
- `table_conv/`: FT-Transformer, TabTransformer, ExcelFormer, SAINT, TransTab, Trompt, and ResNet layers.

Baseline tests live under `test/nn/conv/`, but some use stale constructor names. Confirm current signatures before classifying failures.

### `rllm/nn/models` and `rllm/nn/loss`

- Table models: `TableResNet`, TransTab, and example-assembled pre-encoder/conv/head stacks.
- Graph and relational models: HeteroSAGE, `RDL`, `RelGNNModel`, `BRIDGE`, `InRTL`, and `metartl/`.
- Losses: contrastive losses and VPCL.

Relational failures often span table encoding, sampled indices, and heterogeneous routes. Inspect callers in `examples/rdl.py`, `examples/relgnn.py`, `examples/bridge/`, `examples/inrtl.py`, and `examples/metartl.py` rather than testing only the output head.

### `rllm/dataloader`

- `NeighborLoader`: inbound-neighbor sampling for homogeneous graphs; returns `(batch_size, n_id, adjs)`.
- `HeteroSampler`: disjoint heterogeneous and temporal sampling with pyg-lib or a Python fallback.
- `sampler/data_type.py`: `NodeSamplerInput`, `HeteroSamplerOutput`, and `NumNeighbors` boundary structures.
- `RelbenchLoader`: task rows -> seed entity/time -> sampler output -> filtered stores -> labels, time, and local edges.
- `BRIDGELoader`: BRIDGE-specific loading.

Baseline tests cover `NeighborLoader` and one heterogeneous temporal case. Test `RelbenchLoader` with an in-memory fake dataset, not a real download.

### `rllm/llm`

- `BaseLLM`: `metadata`, `chat`, and `complete` abstractions.
- `LLM` and `LangChainLLM`: general behavior and LangChain adaptation.
- `predictor.py`, `enhancer.py`: prompt/parser and row-enrichment workflows.
- `retrieval/`, `featllm/`, `finetune/`: optional FAISS, HuggingFace, LangChain, and training boundaries.
- `types.py`, `prompt/`, `parser/`: message/response structures and pure prompt parsing.

The baseline has no `test/llm/`. Avoid importing retrieval dependencies when testing pure messages or prompts; mock external adapters instead of issuing requests.

### `rllm/utils`

- `graph_utils.py`, `sparse.py`, `undirected.py`, `seg_reduce.py`: graph indices, sparse formats, undirected conversion, and segmented reduction.
- `_dataloader.py`, `filter_storage.py`: sampled-storage filtering and selection.
- `atomic_routes.py`: RelGNN route construction.
- `download.py`, `extract.py`, `csv_utils.py`: external I/O boundaries.
- `lazy_imports.py`: public lazy exports and optional-dependency collection behavior.

Baseline tests: `test/utils/test_graph_utils.py`; `test/utils/seg_softmax.py` is not collected by default because its filename does not match `test_*.py`.

## Symptom Routing

| Symptom | Inspect first |
| --- | --- |
| `TypeError: unexpected keyword` | Current signature, same-revision example, and whether the test targets an old API |
| Collection-time import error | Lazy-export table, top-level test imports, and eager optional dependencies |
| Wrong column count or order | `TableData.feat_cols`, `col_types.items()`, `df_to_tensor`, metadata order, `TablePreEncoder.forward` |
| Categorical index overflow or label swap | `encode_categorical`, `StatType.COUNT`, embedding offsets, independent target encoding |
| Remaining NaN or `-1` | `fillna.py`, `TableTransform.nan_forward`, encoder padding semantics |
| Slice length or row mismatch | `TableData._tensor_slice`, shared `df`/metadata, `y`, tuple-valued text tensors |
| Wrong graph output rows | `MessagePassing.propagate`, destination `dim_size`, bipartite inputs, isolated destinations |
| Edge direction or local-index error | CSC `row=src,col=dst`, inbound sampling, sampler output, `filter_edge_store_` |
| Temporal leakage | Seed time, CSC time order, `neighbor_time <= seed_time`, label-attachment timing |
| Relational label mismatch | `NodeSamplerInput.input_id`, seed-first ordering, `AttachTargetTransform` |
| Train/eval discrepancy | Dropout, BatchNorm, sampling RNG, and `.training` state before changing tolerance |
| Unexpected network or cache write | `Dataset.__init__`, text-model defaults, example paths, and missing `tmp_path` isolation |
| Wrong rllm checkout imported | `rllm.__file__`, pytest working directory, and `PYTHONPATH` |

Treat every row as a route, not a diagnosis. Establish root cause with a minimal reproduction plus implementation and caller evidence.

