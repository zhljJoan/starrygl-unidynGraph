# StarryUniGraph Library Documentation

## Overview

`StarryUniGraph` is a unified scheduling and runtime library for distributed dynamic graph learning. It exposes one top-level session API for both:

- `CTDG` workloads: continuous-time dynamic graphs, aligned with BTS-MTGNN-style sampling and memory-driven execution.
- `DTDG` workloads: discrete-time dynamic graphs, aligned with Flare-style snapshot/window-driven execution.

The library is organized so that users interact with a single orchestration layer, while the system internally dispatches to graph-family-specific runtime backends.

Current status:

- Unified `session/provider/config/checkpoint` stack is implemented.
- Unified `core/protocol.py` is implemented.
- `CTDG` and `DTDG` family-specific kernels are implemented.
- BTS C++ sampler has been vendored into this repository and a runtime loader is provided.
- CTDG online runtime (`runtime/online/`) is implemented with memory bank, historical cache, sampler, and distributed sync.
- Flare high-performance components are fully merged: `PartitionData`, `STGraphLoader`, `RNNStateManager`, route-aware pinned-memory loading, and CUDA-stream-based async data movement.
- DTDG `flare_native` pipeline is end-to-end operational with DDP training support.

## Design Goals

The library is designed around four goals:

1. Give users one entrypoint for CTDG and DTDG training and inference.
2. Separate orchestration from execution kernels.
3. Preserve family-specific pipelines instead of flattening CTDG and DTDG into one artificial internal algorithm.
4. Allow gradual migration of high-performance backend components from BTS-MTGNN and FlareDTDG into this repository.

## Package Layout

### Top-level package

- `starry_unigraph/__init__.py` — re-exports `SchedulerSession`
- `starry_unigraph/session.py` — unified session lifecycle
- `starry_unigraph/types.py` — shared data types (`DistributedContext`, `SessionContext`, `RuntimeBundle`, `PreparedArtifacts`, `PredictionResult`)
- `starry_unigraph/distributed.py` — distributed init/finalize utilities

### Configuration

- `starry_unigraph/config/default.yaml` — default config values
- `starry_unigraph/config/schema.py` — config loading, merging, validation, graph mode detection

### Registries

- `starry_unigraph/registry/model_registry.py` — maps model name/family to `ModelSpec` and `graph_mode`
- `starry_unigraph/registry/provider_registry.py` — maps `graph_mode` to provider class
- `starry_unigraph/registry/task_registry.py` — maps task type to task adapter

### Core execution

- `starry_unigraph/core/protocol.py` — unified kernel protocol (`KernelBatch`, `KernelExecutor`, `PipelineTrace`, etc.)
- `starry_unigraph/core/ctdg_kernel.py` — CTDG-specific kernel data types and executor
- `starry_unigraph/core/dtdg_kernel.py` — DTDG-specific kernel data types and executor
- `starry_unigraph/core/data_split.py` — train/val/test split ratio normalization and bounds computation

### Data

- `starry_unigraph/data/raw_temporal.py` — raw event loading (`RawTemporalEvents`, CSV/edge-file/mock loaders)
- `starry_unigraph/data/partition.py` — partition data structures (`TensorData`, `RouteData`, `PartitionData`)

### Providers

- `starry_unigraph/providers/common.py` — shared artifact utilities, `BaseProvider` abstract class
- `starry_unigraph/providers/ctdg.py` — `CTDGPreprocessor`, `CTDGProvider`
- `starry_unigraph/providers/dtdg.py` — `DTDGProvider` (dispatches to Flare pipeline)
- `starry_unigraph/providers/dtdg_loaders.py` — `FlareRuntimeLoader` (wraps `STGraphLoader` with split management)
- `starry_unigraph/providers/dtdg_preprocess.py` — `BaseDTDGPreprocessor`, `FlareDTDGPreprocessor`
- `starry_unigraph/providers/dtdg_train.py` — Flare training helpers (`init_flare_training`, `run_flare_train_step`, etc.)

### Preprocessing

- `starry_unigraph/preprocess/base.py` — abstract `GraphPreprocessor`, `ArtifactLayout`, `ArtifactOutput`
- `starry_unigraph/preprocess/dtdg_prepare.py` — DTDG-specific data preparation (snapshot extraction, METIS partitioning, `PartitionData` generation)

### Runtime

- `starry_unigraph/runtime/base.py` — abstract protocols (`RuntimeProtocol`, `RuntimeAdapter`, `ExecutionAdapter`, `GraphProvider`)

#### Flare runtime (`runtime/flare/`)

- `starry_unigraph/runtime/flare/loader.py` — `STGraphLoader` (snapshot iteration with CUDA stream prefetch)
- `starry_unigraph/runtime/flare/state.py` — `RNNStateManager`, `STGraphBlob`, `STWindowState`
- `starry_unigraph/runtime/flare/route.py` — `Route`, `RouteAgent`, differentiable all-to-all exchange with autograd hooks
- `starry_unigraph/runtime/flare/models.py` — Flare DTDG models (`FlareEvolveGCN`, `FlareTGCN`, `FlareMPNNLSTM`, `GCNStack`)
- `starry_unigraph/runtime/flare/training.py` — Flare training/eval/predict step functions with DDP support

#### Online runtime (`runtime/online/`)

- `starry_unigraph/runtime/online/data.py` — `CTDGDataBatch`, `TGTemporalDataset`
- `starry_unigraph/runtime/online/memory.py` — `CTDGMemoryBank` (per-node memory + K-slot mailbox + distributed async sync)
- `starry_unigraph/runtime/online/cache.py` — `CTDGHistoricalCache`, `AdaParameter` (cosine-distance change detection)
- `starry_unigraph/runtime/online/sampler.py` — `NativeTemporalSampler`, `CTDGSampleOutput`
- `starry_unigraph/runtime/online/models.py` — `CTDGMemoryUpdater`, `CTDGLinkPredictor`, `CTDGModelOutput`
- `starry_unigraph/runtime/online/route.py` — `CTDGFeatureRoute`, `AsyncExchangeHandle`
- `starry_unigraph/runtime/online/runtime.py` — `CTDGOnlineRuntime` (full CTDG train/eval/predict orchestration)

### Backends (compatibility shims)

- `starry_unigraph/backends/__init__.py` — re-exports commonly used classes
- `starry_unigraph/backends/flare/__init__.py` — re-exports from `runtime/flare/`
- `starry_unigraph/backends/ctdg/__init__.py` — re-exports from `runtime/online/`

### Tasks

- `starry_unigraph/tasks/base.py` — `BaseTaskAdapter`
- `starry_unigraph/tasks/temporal_link_prediction.py` — `TemporalLinkPredictionTaskAdapter`
- `starry_unigraph/tasks/node_regression.py` — `NodeRegressionTaskAdapter`

### Native and vendored code

- `starry_unigraph/native/bts_sampler.py` — BTS sampler Python wrapper (`BTSNativeSampler`, `build_temporal_neighbor_block`)
- `starry_unigraph/lib/loader.py` — `load_bts_sampler_module()` (lazy `.so` loading via importlib)
- `starry_unigraph/lib/libstarrygl_sampler.so` — precompiled BTS sampler binary
- `starry_unigraph/vendor/bts_sampler/` — BTS C++ source and CMake build

### Checkpoint

- `starry_unigraph/checkpoint/io.py` — `save_checkpoint()`, `load_checkpoint()`

### CLI and tests

- `starry_unigraph/cli/main.py` — CLI entrypoint (prepare/train/predict/resume)
- `tests/test_session.py` — session-level tests

## Core Architecture

The architecture has three main layers plus two runtime backends.

### 1. Session layer

The top-level user interface is `SchedulerSession` in `session.py`.

Main entrypoints:

- `SchedulerSession.from_config(...)` — load config, infer graph mode, construct provider
- `prepare_data()` — trigger preprocessing, write artifacts
- `build_runtime()` — initialize model, optimizer, loaders, runtime state
- `run_epoch(split)` — iterate batches for one epoch
- `run_task()` — full train/eval loop across epochs
- `evaluate(split)` — single evaluation pass
- `predict(split)` — inference pass returning `PredictionResult`
- `save_checkpoint(path)` / `load_checkpoint(path)` — persist/restore state

`SchedulerSession` is responsible for:

- loading and validating configuration
- resolving model family and graph mode
- constructing the correct provider
- running unified train/eval/predict/checkpoint flows

It is intentionally not responsible for implementing graph-family-specific execution details.

### 2. Provider layer

Providers map config and artifacts to runtime backends.

- `providers/ctdg.py` — orchestrates via `CTDGOnlineRuntime`
- `providers/dtdg.py` — orchestrates via `FlareRuntimeLoader` + Flare training steps

Provider responsibilities:

- prepare artifact directories and metadata
- instantiate family-specific partition/route/runtime objects
- expose iterators to the session layer
- delegate step execution to runtime backends

Provider non-responsibilities:

- sampling implementation (belongs in `runtime/online/`)
- snapshot execution (belongs in `runtime/flare/`)
- memory update logic (belongs in `runtime/online/`)

### 3. Kernel layer

The kernel layer defines execution protocol and family-specific data types.

- `core/protocol.py` defines the unified execution protocol.
- `core/ctdg_kernel.py` defines CTDG batch/state/result types.
- `core/dtdg_kernel.py` defines DTDG batch/state/result types.

### 4. Runtime backends

The actual execution logic lives in two runtime backends:

- `runtime/flare/` — Flare-style DTDG execution (snapshot loading, GCN/RNN models, route exchange, DDP training)
- `runtime/online/` — BTS-style CTDG execution (temporal sampling, memory bank, historical cache, distributed sync)

These backends are the layer where pipeline semantics live. The `backends/` package provides backward-compatible re-exports.

## Unified Protocol Layer

The protocol is defined in `core/protocol.py`.

Key abstractions:

- `KernelBatch` — base for batch types (has `index`, `split`, `chain`, `to_payload()`)
- `KernelRuntimeState` — execution state that evolves across steps
- `KernelResult` — step output (`predictions`, `targets`, `loss`, `meta`)
- `KernelExecutor` — common execution interface (`iter_batches`, `execute_train/eval/predict`, `dump_state`)
- `PipelineTrace` — records stage-level execution with async-flow management
- `AsyncStageHandle` — async stage tracking (`token`, `name`, `status`, `depends_on`)
- `StateHandle` — identifies state container (CTDG: node memory; DTDG: window state)
- `StateDelta` — describes state change produced by a step
- `StateWriteback` — combines `StateHandle` + `StateDelta` + version

Pipeline trace stages:

- CTDG: `sample`, `feature_fetch`, `state_fetch`, `memory_updater`, `neighbor_attention_aggregate`, `message_generate`, `state_transition`, `state_writeback`
- DTDG: `load_snapshot`, `route_apply`, `state_fetch`, `state_transition`, `state_writeback`

## CTDG Library Stack

### Core kernel types (`core/ctdg_kernel.py`)

- `FeatureRoutePlan` — route_type, fanout, feature_keys
- `StateSyncPlan` — sync mode and version counter
- `CTDGPartitionBook` — partition metadata
- `CTDGMailboxState` — memory/mailbox versioning
- `CTDGBatch(KernelBatch)` — batch with sampling plan
- `CTDGPreparedBatch` — materialized batch
- `CTDGStepResult(KernelResult)` — step output
- `CTDGRuntimeState(KernelRuntimeState)` — runtime cursor

### Online runtime (`runtime/online/`)

The CTDG execution path is implemented in `runtime/online/`:

**Data pipeline:**

- `TGTemporalDataset` — loads raw events, provides distributed-aware batched iteration with train/val/test splits
- `CTDGDataBatch` — mini-batch of events (src, dst, ts, edge_feat)

**Sampling:**

- `NativeTemporalSampler` — wraps vendored BTS C++ sampler for multi-hop temporal neighbor sampling
- `CTDGSampleOutput` — sampled neighbors, edges, timestamps

**Memory and state:**

- `CTDGMemoryBank` — per-node hidden memory + K-slot mailbox with distributed async sync via `all_to_all_single`
- `CTDGHistoricalCache` — caches last-synced memory, uses cosine-distance filtering (`AdaParameter`) to skip redundant syncs

**Models:**

- `CTDGMemoryUpdater` — GRU-based memory update over mailbox history
- `CTDGLinkPredictor` — temporal transformer attention for link prediction

**Routing:**

- `CTDGFeatureRoute` — inter-partition feature exchange
- `AsyncExchangeHandle` — async send/recv handles

**Orchestration:**

- `CTDGOnlineRuntime` — full CTDG train/eval/predict loop (memory update -> sampling -> conv -> scoring -> distributed sync)

### CTDG execution pipeline

The CTDG runtime preserves the BTS-style stage order:

1. `sample` — temporal neighbor sampling via native BTS sampler
2. `feature_fetch` — route-aware feature retrieval
3. `state_fetch` — gather memory/mailbox state
4. `memory_updater` — GRU update over mailbox
5. `neighbor_attention_aggregate` — attention over sampled temporal neighborhood
6. `message_generate` — produce messages for state update
7. `state_transition` — update node memory
8. `state_writeback` — write back to distributed memory bank

### CTDG artifact model

The CTDG provider writes:

- `artifacts/{dataset}/meta/artifacts.json`
- `artifacts/{dataset}/partitions/manifest.json`
- `artifacts/{dataset}/routes/manifest.json`
- `artifacts/{dataset}/sampling/index.json`

### BTS native sampler integration

- vendored source: `vendor/bts_sampler/` (C++ with CMake)
- precompiled binary: `lib/libstarrygl_sampler.so`
- runtime loader: `lib/loader.py` (`load_bts_sampler_module()`)
- Python wrapper: `native/bts_sampler.py` (`BTSNativeSampler`, `build_temporal_neighbor_block`, `is_bts_sampler_available`)

The native sampler is attached via `CTDGSamplerCore.attach_native_sampler(...)` and consumed by `NativeTemporalSampler` in the online runtime.

## DTDG Library Stack

### Core kernel types (`core/dtdg_kernel.py`)

- `SnapshotRoutePlan` — route_type, cache_policy
- `DTDGPartitionBook` — partition metadata with snapshot count
- `DTDGWindowState` — window tracking
- `DTDGBatch(KernelBatch)` — batch with adjacency, features, graph, and graph_meta
- `DTDGStepResult(KernelResult)` — step output
- `DTDGRuntimeState(KernelRuntimeState)` — snapshot cursor

### Data structures (`data/partition.py`)

- `TensorData` — packed variable-length tensor storage (CSR-style ptr/ind)
- `RouteData` — per-snapshot routing descriptors (send/recv sizes, send_index)
- `PartitionData` — complete partition container (node_data, edge_data, labels, routes, chunks). Supports `save()`/`load()`, `pin_memory()`, `to(device)`, `to_blocks()` conversion to DGL blocks.

### Flare runtime (`runtime/flare/`)

The DTDG execution path is fully implemented in `runtime/flare/`:

**Data loading:**

- `STGraphLoader` — iterates over `PartitionData` snapshots as DGL blocks. Uses `pin_memory()` + dedicated `torch.cuda.Stream` for async CPU-to-GPU data transfer. Factory: `STGraphLoader.from_partition_data(data, device)`.

**State management:**

- `RNNStateManager` — manages sliding window of graphs with per-snapshot RNN state. Patches each graph with `flare_fetch_state()` / `flare_store_state()` methods. Supports "pad" mode (zero-pad new state) and "mix" mode (blend with previous).
- `STGraphBlob` — wraps a sequence of graphs + `RNNStateManager`. Iterable. Property `current_graph` returns the last graph in the window.
- `STWindowState` — window size and snapshot tracking metadata.

**Routing:**

- `Route` — per-snapshot differentiable all-to-all feature exchange. Autograd-compatible via `RouteSendFunction` / `RouteRecvFunction`.
- `RouteAgent` — executes actual `dist.all_to_all_single` communication.

**Models:**

- `GCNStack` — multi-layer GCN message passing on DGL blocks
- `FlareEvolveGCN` — EvolveGCN-H (GRU-evolving GCN weights per snapshot)
- `FlareTGCN` — Temporal GCN (GCN + GRU per snapshot)
- `FlareMPNNLSTM` — MPNN-LSTM (GCN + two custom `_LSTMCell` layers). State is a 4-tuple `(h1,c1,h2,c2)` via Python tuple concatenation.
- `build_flare_model(name, ...)` — factory function

**Training:**

- `init_flare_training(config, partition_data)` — builds model, optimizer, DDP wrapping, computes per-partition loss scale
- `run_flare_train_step(blob, model, optimizer, ...)` — forward + backward + optimizer step
- `run_flare_eval_step(blob, model, ...)` — forward (no grad)
- `run_flare_predict_step(blob, model, ...)` — forward (no grad), returns predictions

### Provider integration (`providers/dtdg*.py`)

- `FlareDTDGPreprocessor` — generates `PartitionData` files (`part_NNN.pth`) via METIS partitioning and snapshot extraction
- `FlareRuntimeLoader` — wraps `STGraphLoader` with train/eval/predict split management. Training yields `STGraphBlob`; eval/predict yields `DTDGBatch` (single frame).
- `DTDGProvider` — orchestrates the Flare pipeline end-to-end: preprocessing -> runtime build -> iterator construction -> step execution

### DTDG execution pipeline

The Flare-native pipeline preserves the following stage order:

1. `load_snapshot` — `STGraphLoader` fetches and transfers snapshot to GPU via CUDA stream
2. `route_apply` — `Route.forward()` performs differentiable all-to-all feature exchange
3. `state_fetch` — `RNNStateManager` provides per-snapshot RNN state via `flare_fetch_state()`
4. `state_transition` — model forward pass (GCN + RNN layers)
5. `state_writeback` — `flare_store_state()` writes updated RNN state back to manager

### DTDG artifact model

The DTDG provider writes:

- `artifacts/{dataset}/meta/artifacts.json`
- `artifacts/{dataset}/partitions/manifest.json`
- `artifacts/{dataset}/routes/manifest.json`
- `artifacts/{dataset}/flare/manifest.json`, `flare/part_NNN.pth` (Flare-native)
- `artifacts/{dataset}/snapshots/manifest.json` (chunked, if applicable)

### DTDG preprocessing (`preprocess/dtdg_prepare.py`)

Full DTDG data preparation pipeline:

- `build_partition_input_from_raw_dataset()` — extracts snapshots and metadata from raw events
- `build_dtdg_partitions()` — runs METIS or random graph partitioning
- `build_flare_partition_data_list()` — creates per-partition `PartitionData` objects with local edge indices, node features, labels, and route descriptors

### Data split utilities (`core/data_split.py`)

- `normalize_split_ratio(split_ratio)` — ensures train+val+test sum to 1.0
- `split_bounds(total, split, split_ratio)` — returns `(start, end)` index range for a given split

## Configuration System

The configuration system is defined in `config/schema.py` with defaults in `config/default.yaml`.

### Main sections

- `model` — name, family, task, hidden_dim, memory (CTDG), window (DTDG)
- `data` — root, name, format, graph_mode, split_ratio
- `train` — epochs, batch_size, snaps, eval_interval, lr
- `runtime` — backend, device, cache, checkpoint
- `dtdg` — pipeline, chunk_order, chunk_decay, num_full_snaps
- `ctdg` — pipeline, mailbox_slots, historical_alpha, async_sync, ada_param_enabled, dropout
- `preprocess` — cluster and chunk settings
- `sampling` — neighbor_limit, strategy, history, neg_sampling
- `graph` — storage, partition, route
- `dist` — backend, world_size, rank, local_rank, master_addr, master_port

### Graph mode dispatch

`model.family` is resolved through the model registry:

- CTDG families: `tgn`, `dyrep`, `jodie`, `tgat`, `apan`
- DTDG families: `evolvegcn`, `tgcn`, `mpnn_lstm`, `gcn`

This drives provider resolution and kernel selection.

### Validation behavior

The schema layer:

- fills defaults from `default.yaml`
- validates required config paths (`model.name`, `data.root`, etc.)
- infers graph mode when possible
- warns about inactive fields (e.g., CTDG-specific fields when running DTDG)
- raises hard errors for missing required fields

## Registry System

The registry layer decouples names from implementations.

### ModelRegistry

Maps `model.name` or `model.family` to `ModelSpec` (containing name, family, graph_mode).

### ProviderRegistry

Maps `ctdg` to `CTDGProvider`, `dtdg` to `DTDGProvider`.

### TaskRegistry

Maps task types to task adapter classes:

- `"temporal_link_prediction"` -> `TemporalLinkPredictionTaskAdapter`
- `"snapshot_node_regression"` -> `NodeRegressionTaskAdapter`

## Runtime and Checkpointing

The runtime abstraction lives in `runtime/base.py` with the following abstract protocols:

- `RuntimeProtocol` — `iter_batches()`, `step()`, `state_dict()`, `load_state_dict()`
- `RuntimeAdapter` — model/optimizer init, runtime state load/dump
- `ExecutionAdapter` — `train_step()`, `eval_step()`, `predict_step()`
- `GraphProvider` — combines adapter + execution with `graph_mode` property

Checkpoint I/O lives in `checkpoint/io.py`.

Current checkpoint payload:

- `model_state`
- `optimizer_state`
- `scheduler_state`
- `runtime_state`
- `provider_meta`
- `config`
- `epoch`
- `global_step`

Runtime state includes provider-mapped state:

- CTDG: `memory_state`, `mailbox_state`, `sampler_state`, `executor_state`
- DTDG: `window_state`, `snapshot_state`, `route_cache`, `executor_state`

## Preprocessing Layer

The abstract preprocessing contract is defined in `preprocess/base.py`:

- `ArtifactLayout` — root directory + subdirectory mapping
- `ArtifactOutput` — relative_path, payload, serializer (json/torch)
- `ArtifactPayload` — provider_meta + outputs list
- `GraphPreprocessor` (abstract) — `prepare_raw()`, `build_partitions()`, `build_runtime_artifacts()`, `run()`

Concrete preprocessors:

- `CTDGPreprocessor` — prepares raw temporal events into partition manifest + feature route plan
- `BaseDTDGPreprocessor` / `FlareDTDGPreprocessor` — full DTDG preprocessing pipeline: raw event loading -> snapshot extraction -> METIS partitioning -> `PartitionData` generation with `part_NNN.pth` files

DTDG data preparation (`preprocess/dtdg_prepare.py`) provides:

- snapshot dataset construction from raw events
- lifetime graph analysis for partitioning
- METIS/random partition algorithm
- per-partition `PartitionData` assembly with local edges, features, labels, and routes

## Distributed Support

Distributed utilities live in `distributed.py`:

- `apply_distributed_env()` — reads `WORLD_SIZE`/`RANK`/`LOCAL_RANK` from environment (set by `torchrun`), merges into config
- `build_distributed_context()` — creates `DistributedContext` from config
- `initialize_distributed()` — calls `dist.init_process_group()`, sets CUDA device
- `finalize_distributed()` — calls `dist.destroy_process_group()`

DDP training is implemented in:

- Flare: `init_flare_training()` wraps model in `DistributedDataParallel`, computes per-partition loss scale via `all_reduce`
- Online: `CTDGOnlineRuntime` handles distributed memory sync via `all_to_all_single`

## CLI

CLI entrypoint is `cli/main.py`.

Commands:

- `prepare` — run preprocessing only
- `train` — prepare + build + train
- `predict` — run inference
- `resume` — restore from checkpoint and continue

Examples:

```bash
python -m starry_unigraph.cli prepare --config path/to/config.yaml
python -m starry_unigraph.cli train --config path/to/config.yaml
python -m starry_unigraph.cli predict --config path/to/config.yaml --split test
python -m starry_unigraph.cli resume --config path/to/config.yaml --checkpoint path/to/ckpt.pkl
```

## Current Execution Flow

### Session creation

```python
from starry_unigraph import SchedulerSession

session = SchedulerSession.from_config("starry_unigraph/config/default.yaml")
```

### Prepare and build

```python
session.prepare_data()
session.build_runtime()
```

### Train

```python
summary = session.run_epoch(split="train")
```

### Full training loop

```python
result = session.run_task()
```

### Predict

```python
result = session.predict(split="test")
```

### Save and restore

```python
session.save_checkpoint("checkpoint.pkl")
session.load_checkpoint("checkpoint.pkl")
```

## Pipeline Semantics

One of the central design principles is that the system should not erase the original family-specific execution model.

### CTDG pipeline semantics

The library preserves:

- sampling-first execution via native BTS sampler
- route-aware feature fetch across partitions
- memory/mailbox update with distributed async sync
- historical cache with adaptive cosine-distance filtering

### DTDG pipeline semantics

The library preserves:

- snapshot loading with CUDA stream prefetch
- differentiable route-based all-to-all feature exchange
- RNN state management with sliding window (pad/mix modes)
- per-partition loss scaling for balanced DDP training

## What Is Already Merged From BTS and Flare

### Already merged

- Unified config/session/provider/checkpoint architecture
- BTS sampler source and binary assets
- BTS sampler Python wrapper and loader
- Unified kernel protocol
- CTDG kernel with BTS-style stage structure
- CTDG online runtime with memory bank, historical cache, sampler, and distributed sync
- DTDG kernel with Flare-style stage structure
- Flare `PartitionData` with CSR-style tensor storage
- Flare `STGraphLoader` with CUDA stream prefetch
- Flare `RNNStateManager` and `STGraphBlob`
- Flare differentiable `Route` with autograd hooks
- Flare DTDG models (`FlareEvolveGCN`, `FlareTGCN`, `FlareMPNNLSTM`)
- Flare DDP training with per-partition loss scaling
- DTDG preprocessing with METIS partitioning and `PartitionData` generation

### Not fully merged yet

- Full BTS native sampler-driven end-to-end CTDG training path (sampler is integrated but kernel auto-dispatch is partial)
- BTS memory/mailbox full implementation parity with upstream
- DTDG `chunked` pipeline high-performance backend (adaptive chunk prepare exists but is not yet fully integrated)

## Testing

Tests are currently in `tests/test_session.py`.

Covered behaviors:

- provider resolution from config
- artifact generation
- runtime build
- CTDG and DTDG pipeline execution
- checkpoint save/load
- prediction output shape
- inactive-field warnings
- missing config path failure
- artifact version mismatch failure
- missing route manifest failure
- BTS native sampler module availability

## Summary

`StarryUniGraph` is a unified dynamic graph library with:

- one public session API
- one config system
- one checkpoint format
- one execution protocol
- two family-specific runtime backends (Flare for DTDG, Online for CTDG)
- full Flare high-performance data path (PartitionData, STGraphLoader, RNNStateManager, differentiable routing, DDP training)
- CTDG online runtime with memory bank, sampling, and distributed sync

The library provides structural unification while preserving original pipeline semantics for both CTDG and DTDG workloads.
