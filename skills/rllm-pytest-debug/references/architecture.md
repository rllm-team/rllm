# Architecture and Execution Paths

Read this reference for tests spanning data objects, encoders, samplers, and models, or when selecting an impact-aware regression scope.

## Analysis Baseline

- `SOURCE_COMMIT`: `7421bf22491516119d7d57fcb67019d9ef02164a`
- Commit subject: `Merge pull request #309 from rllm-team/inrtl`
- Analysis date: 2026-09-12
- Repository: `rllm-team/rllm`

The review scanned all 289 tracked paths and Python symbols at the baseline, then examined:

- `rllm/types.py` and the table, graph, heterogeneous storage, preprocessing, and transform implementations;
- column encoders, `TablePreEncoder`, representative table layers, `MessagePassing`, GCN/GAT, and temporal encoders;
- `NeighborLoader`, `HeteroSampler`, the Python sampler fallback, sampler data types, `RelbenchLoader`, and storage filtering;
- `TableResNet`, `RDL`, `RelGNNModel`, `InRTL`, and representative MetaRTL components;
- LLM messages, base/general/LangChain adapters, predictor, enhancer, and retrieval boundaries;
- all baseline tests, representative examples, Sphinx entry points, packaging files, and GitHub Actions.

This was not a line-by-line audit of every dataset or model variant. Repeated table and graph layers were grouped by abstraction; InRTL and MetaRTL were read through their main composition paths without real-data training.

## Layering

1. `ColType`, `TableType`, `TaskType`, and `StatType` describe raw columns and task semantics.
2. `TableData`, `GraphData`, and `HeteroGraphData` package DataFrames, typed tensors, edges, and custom attributes with device transfer, persistence, and copy behavior.
3. Preprocessors materialize `feat_dict` and metadata; transforms compose mutations over data objects or tensors.
4. `ColEncoder -> TablePreEncoder -> table layer/model` handles tables; `MessagePassing -> graph layer/model` handles graphs.
5. Loaders and samplers preserve alignment among global entity IDs, local batch IDs, time, and labels. Relational models encode each table before heterogeneous propagation.
6. Datasets and examples connect abstractions to data, training, losses, and metrics. The LLM subsystem is a separate, optional-dependency-heavy table enrichment and prediction path.

Several subpackages use `define_lazy_imports` for public symbols. Test both the concrete module and the lazy export when diagnosing import boundaries.

## Method Families and Representative Paths

rllm contains multiple methods in each family:

- **Tabular neural networks:** FT-Transformer, TabTransformer, ExcelFormer, SAINT, TransTab, Trompt, and ResNet paths. FT-Transformer is the representative path below.
- **Graph neural networks:** GCN, GAT, GraphSAGE, LGC, HAN, HGT, graph Transformer, and related layers. Homogeneous neighbor sampling plus GCN is the representative path.
- **Relational table learning:** RDL, RelGNN, BRIDGE, InRTL, and MetaRTL. RelBench plus RelGNN is the representative path.

These families overlap: relational methods reuse table encoders, heterogeneous graph storage, and message passing. The paths below are examples, not exhaustive model definitions. For a directory-level task, scan the actual implementations and select tests across shared abstractions and distinct designs.

## Representative Path 1: Tabular Learning, e.g. FT-Transformer

Entry point: `examples/ft_transformer.py`. Core symbols: `TableData`, `df_to_tensor`, `DefaultTableTransform`, `FTTransformerPreEncoder`, `FTTransformerConv`.

1. A dataset supplies a DataFrame and ordered `col_types`; `TableData` normalizes the primary key and, unless lazy, materializes features and metadata.
2. `df_to_tensor()` follows `col_types.items()`. Numerical values become float32 `[N, 1]`; each categorical column is factorized to integer features; the target is excluded from features and becomes `y`. Groups are `[N, C]`, embedded text is `[N, C, D_text]`, and tokenized text is two `[N, C, L]` tensors.
3. `DefaultTableTransform.forward()` passes through `TableTransform.nan_forward()` and replaces `TableData.feat_dict`; this is observable state mutation.
4. `TableData.get_dataloader()` yields `(feat_dict, y)` batches whose keys are `ColType` values.
5. `FTTransformerPreEncoder` uses `EmbeddingEncoder` for categorical and `LinearEncoder` for numerical data. `TablePreEncoder.forward()` produces `[B, C_type, H]` per type and concatenates them in `feat_dict` iteration order to `[B, C_total, H]`.
6. `FTTransformerConv` prepends a `[B, 1, H]` CLS token and returns `[B, H]` when `use_cls=True`, otherwise `[B, C_total, H]`. The example's choice to take the first column when `use_cls=False` is model-specific.
7. Classification callers cast `y.long()` for cross entropy; materialized targets may still be float32.

Protect column/metadata order, per-column statistics, and batch-label alignment. Relevant tests are under `test/data/`, `test/nn/encoder/`, and `test/nn/conv/`; the baseline FT test uses a stale constructor parameter.

Permanent source links: [TableData](https://github.com/rllm-team/rllm/blob/7421bf22491516119d7d57fcb67019d9ef02164a/rllm/data/table_data.py#L91), [df_to_tensor](https://github.com/rllm-team/rllm/blob/7421bf22491516119d7d57fcb67019d9ef02164a/rllm/preprocessing/df_to_tensor.py#L206), [TablePreEncoder](https://github.com/rllm-team/rllm/blob/7421bf22491516119d7d57fcb67019d9ef02164a/rllm/nn/encoder/table_pre_encoder.py#L12).

## Representative Path 2: Graph Learning, e.g. Neighbor Sampling and GCN

Entry points: `examples/gcn.py`, `examples/gcn_batch.py`, and `NeighborLoader`.

1. `PlanetoidDataset` stores node features `[N, F]`, labels `[N]`, sparse adjacency `[N, N]`, and boolean train/validation/test masks in `GraphData`.
2. The full-graph path applies `GCNTransform`: `NormalizeFeatures` updates node features, while `GCNNorm` adds remaining self-loops and symmetric normalization. `GCNNorm.forward` is cached.
3. `GCNConv.forward` applies a linear map and then `MessagePassing.propagate`. For edge lists, row 0 is source and row 1 is destination; destination `dim_size` controls output rows.
4. `NeighborLoader` builds destination-sorted CSC-like pointers from an edge list or sparse indices. It samples inbound source neighbors, keeps seeds first in `n_id`, and appends newly found nodes.
5. Global IDs are mapped to local `n_id` positions for each-hop sparse adjacency. The loader returns `(batch_size, n_id, adjs)`; multi-layer example code consumes adjacencies in its documented reverse order.

Protect edge direction, destination counts, sparse layout, global/local ID mapping, and isolated-destination output rows. Empty-hop tensors are current behavior, not proof that every model supports every empty layout. Relevant tests are `test/utils/`, `test/nn/conv/`, and `test/dataloader/`; the baseline loader test targets an old API.

Permanent source links: [NeighborLoader](https://github.com/rllm-team/rllm/blob/7421bf22491516119d7d57fcb67019d9ef02164a/rllm/dataloader/neighbor_loader.py#L9), [MessagePassing](https://github.com/rllm-team/rllm/blob/7421bf22491516119d7d57fcb67019d9ef02164a/rllm/nn/conv/graph_conv/message_passing.py#L14), [GCNConv](https://github.com/rllm-team/rllm/blob/7421bf22491516119d7d57fcb67019d9ef02164a/rllm/nn/conv/graph_conv/gcn_conv.py#L13).

## Representative Path 3: Relational Learning, e.g. RelBench to RelGNN

Entry point: `examples/relgnn.py`. Core boundaries: `RelBenchDataset`, `HeteroSampler`, `RelbenchLoader`, `RelGNNModel`.

1. `make_pkey_fkey_graph()` materializes each `TableData` and creates forward FK-row-to-PK-entity edges plus explicit reverse edges.
2. A task split supplies entity, time, and target columns. `RelbenchLoader._get_sampler_input_from_task()` forms `NodeSamplerInput`; slice-derived `input_id` later selects split targets.
3. `HeteroSampler` converts edge types to CSC. The Python backend creates a disjoint subgraph per seed and filters future source nodes by seed time. pyg-lib and fallback behavior require separate validation.
4. `HeteroSamplerOutput.node` contains global node IDs; `row` and `col` are local batch indices. `_filter_hetero_data()` shallow-copies the graph container, slices table/time stores by global IDs, and writes local edges.
5. The loader adds `n_id`, `input_id`, `seed_time`, `batch_size`, time/batch dictionaries, and edge dictionaries. `AttachTargetTransform` indexes labels by `input_id`, because one entity may appear in multiple task rows at different times.
6. `RelGNNModel.forward()` encodes each node type with `TableResNet`, optionally adds relative-time embeddings, propagates along atomic routes, selects the seed prefix of the target table, and emits `[B, out_dim]`.
7. The example uses BCE-with-logits for binary classification and L1 for regression. Time filtering and label visibility are data contracts, not details to simplify away.

Protect the distinction between entity ID and task-row `input_id`, seed-first ordering, local `row/col`, typed edge direction, and time-to-batch alignment. The baseline directly covers only one temporal sampling case; test loader alignment with an in-memory fake dataset before model integration or real Rel-F1 data.

Permanent source links: [RelBench graph construction](https://github.com/rllm-team/rllm/blob/7421bf22491516119d7d57fcb67019d9ef02164a/rllm/datasets/relbench/base.py#L245), [RelbenchLoader](https://github.com/rllm-team/rllm/blob/7421bf22491516119d7d57fcb67019d9ef02164a/rllm/dataloader/relbench_loader.py#L32), [RelGNNModel](https://github.com/rllm-team/rllm/blob/7421bf22491516119d7d57fcb67019d9ef02164a/rllm/nn/models/relgnn.py#L217).

## Shared Contracts vs. Model-Specific Behavior

- `[B, C, H]` from `TablePreEncoder` is shared; CLS handling and selecting a particular column are model choices.
- Source/destination semantics and `dim_size` belong to `MessagePassing`; GAT attention, GCN normalization, and RelGNN routes are specific implementations.
- Typed storage and `(src, rel, dst)` keys belong to `HeteroGraphData`; task-row time and labels are RelBench-specific.
- Automatic dataset download/process is base-class behavior; URLs, cache names, and splits are dataset-specific.
- `BaseLLM.chat/complete` express an abstract return contract; LangChain, retrieval, FeatLLM, and fine-tuning have distinct third-party boundaries.

## Regression Scope

- Column types, tensorization, or metadata: table pre-encoders, table models, per-table relational encoding, and cache compatibility.
- Storage copy/apply/slice: graph containers, loader filtering, device moves, and persistence.
- Edge direction, CSC, or local IDs: both samplers, mini-batch GNNs, and temporal filtering.
- Lazy exports: collection-time optional-dependency behavior.
- RelBench task labels: split, time, entity IDs, seed order, and loss dtype/shape.

Use this scope to select tests and explain impact. It does not authorize source changes.
