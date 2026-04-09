# StarryUniGraph Library Documentation

## Overview

`StarryUniGraph` is a unified scheduling and runtime library for distributed dynamic graph learning. It exposes one top-level session API for both:

- `CTDG` workloads: continuous-time dynamic graphs, currently aligned with BTS-MTGNN-style sampling and memory-driven execution.
- `DTDG` workloads: discrete-time dynamic graphs, currently aligned with Flare-style snapshot/window-driven execution.

The library is organized so that users interact with a single orchestration layer, while the system internally dispatches to graph-family-specific kernels.

Current status:

- Unified `session/provider/config/checkpoint` stack is implemented.
- Unified `core/protocol.py` is implemented.
- `CTDG` and `DTDG` family-specific kernels are implemented.
- BTS C++ sampler has been vendored into this repository and a runtime loader is provided.
- Flare high-performance `PartitionData/STGraphLoader` pipeline has not been merged yet.

## Design Goals

The library is designed around four goals:

1. Give users one entrypoint for CTDG and DTDG training and inference.
2. Separate orchestration from execution kernels.
3. Preserve family-specific pipelines instead of flattening CTDG and DTDG into one artificial internal algorithm.
4. Allow gradual migration of high-performance backend components from BTS-MTGNN and FlareDTDG into this repository.

## Package Layout

### Top-level package

- [starry_unigraph/__init__.py](/home/zlj/StarryUniGraph/starry_unigraph/__init__.py)
- [starry_unigraph/session.py](/home/zlj/StarryUniGraph/starry_unigraph/session.py)
- [starry_unigraph/types.py](/home/zlj/StarryUniGraph/starry_unigraph/types.py)

### Configuration

- [starry_unigraph/config/default.yaml](/home/zlj/StarryUniGraph/starry_unigraph/config/default.yaml)
- [starry_unigraph/config/schema.py](/home/zlj/StarryUniGraph/starry_unigraph/config/schema.py)

### Registries

- [starry_unigraph/registry/model_registry.py](/home/zlj/StarryUniGraph/starry_unigraph/registry/model_registry.py)
- [starry_unigraph/registry/provider_registry.py](/home/zlj/StarryUniGraph/starry_unigraph/registry/provider_registry.py)
- [starry_unigraph/registry/task_registry.py](/home/zlj/StarryUniGraph/starry_unigraph/registry/task_registry.py)

### Core execution

- [starry_unigraph/core/protocol.py](/home/zlj/StarryUniGraph/starry_unigraph/core/protocol.py)
- [starry_unigraph/core/ctdg_kernel.py](/home/zlj/StarryUniGraph/starry_unigraph/core/ctdg_kernel.py)
- [starry_unigraph/core/dtdg_kernel.py](/home/zlj/StarryUniGraph/starry_unigraph/core/dtdg_kernel.py)
- [starry_unigraph/core/ctdg.py](/home/zlj/StarryUniGraph/starry_unigraph/core/ctdg.py)
- [starry_unigraph/core/dtdg.py](/home/zlj/StarryUniGraph/starry_unigraph/core/dtdg.py)

### Providers

- [starry_unigraph/providers/common.py](/home/zlj/StarryUniGraph/starry_unigraph/providers/common.py)
- [starry_unigraph/providers/ctdg.py](/home/zlj/StarryUniGraph/starry_unigraph/providers/ctdg.py)
- [starry_unigraph/providers/dtdg.py](/home/zlj/StarryUniGraph/starry_unigraph/providers/dtdg.py)

### Tasks

- [starry_unigraph/tasks/base.py](/home/zlj/StarryUniGraph/starry_unigraph/tasks/base.py)
- [starry_unigraph/tasks/temporal_link_prediction.py](/home/zlj/StarryUniGraph/starry_unigraph/tasks/temporal_link_prediction.py)
- [starry_unigraph/tasks/node_regression.py](/home/zlj/StarryUniGraph/starry_unigraph/tasks/node_regression.py)

### Native and vendored code

- [starry_unigraph/native/bts_sampler.py](/home/zlj/StarryUniGraph/starry_unigraph/native/bts_sampler.py)
- [starry_unigraph/lib/loader.py](/home/zlj/StarryUniGraph/starry_unigraph/lib/loader.py)
- [starry_unigraph/lib/libstarrygl_sampler.so](/home/zlj/StarryUniGraph/starry_unigraph/lib/libstarrygl_sampler.so)
- [starry_unigraph/vendor/bts_sampler/CMakeLists.txt](/home/zlj/StarryUniGraph/starry_unigraph/vendor/bts_sampler/CMakeLists.txt)

### Runtime and checkpoint

- [starry_unigraph/runtime/base.py](/home/zlj/StarryUniGraph/starry_unigraph/runtime/base.py)
- [starry_unigraph/checkpoint/io.py](/home/zlj/StarryUniGraph/starry_unigraph/checkpoint/io.py)

### CLI and tests

- [starry_unigraph/cli/main.py](/home/zlj/StarryUniGraph/starry_unigraph/cli/main.py)
- [tests/test_session.py](/home/zlj/StarryUniGraph/tests/test_session.py)

## Core Architecture

The current architecture has three main layers.

### 1. Session layer

The top-level user interface is `SchedulerSession` in [session.py](/home/zlj/StarryUniGraph/starry_unigraph/session.py).

Main entrypoints:

- `SchedulerSession.from_config(...)`
- `prepare_data()`
- `build_runtime()`
- `train_step()`
- `run_epoch()`
- `run_task()`
- `predict()`
- `save_checkpoint()`
- `load_checkpoint()`

`SchedulerSession` is responsible for:

- loading and validating configuration
- resolving model family and graph mode
- constructing the correct provider
- running unified train/eval/predict/checkpoint flows

It is intentionally not responsible for implementing graph-family-specific execution details.

### 2. Provider layer

Providers map config and artifacts to kernels.

- [providers/ctdg.py](/home/zlj/StarryUniGraph/starry_unigraph/providers/ctdg.py)
- [providers/dtdg.py](/home/zlj/StarryUniGraph/starry_unigraph/providers/dtdg.py)

Provider responsibilities:

- prepare artifact directories and metadata
- instantiate family-specific partition/route/runtime objects
- construct the correct kernel from config
- expose iterators to the session layer
- map kernel state into unified runtime/checkpoint state

Provider non-responsibilities:

- sampling implementation
- snapshot execution implementation
- memory update logic
- sparse operator execution logic

Those belong in `core`.

### 3. Kernel layer

The kernel layer is the actual execution layer.

- `core/protocol.py` defines the unified execution protocol.
- `core/ctdg_kernel.py` implements the CTDG kernel.
- `core/dtdg_kernel.py` implements the DTDG kernel.

This is the layer where pipeline semantics live.

## Unified Protocol Layer

The protocol is defined in [protocol.py](/home/zlj/StarryUniGraph/starry_unigraph/core/protocol.py).

Key abstractions:

- `KernelBatch`
- `KernelRuntimeState`
- `KernelResult`
- `KernelExecutor`
- `PipelineTrace`
- `AsyncStageHandle`
- `StateHandle`
- `StateDelta`
- `StateWriteback`

### KernelBatch

Represents one unit of work.

Shared assumptions:

- has `index`
- has `split`
- has `chain`
- can serialize itself with `to_payload()`

### KernelRuntimeState

Represents execution-state that evolves across steps.

Examples:

- sampler cursor for CTDG
- snapshot cursor for DTDG
- last prediction
- last active split

### KernelResult

Represents one step result.

Shared payload shape:

- `predictions`
- `targets` when available
- `loss` when available
- `meta`

### KernelExecutor

Defines the common execution interface:

- `iter_batches(split, count)`
- `execute_train(batch)`
- `execute_eval(batch)`
- `execute_predict(batch)`
- `dump_state()`

This is the main internal unification point between BTS-style and Flare-style execution.

### PipelineTrace

`PipelineTrace` records stage-level information while preserving the original family-specific pipeline shape.

It now also carries async-flow management semantics:

- stage ordering for synchronous steps
- async stage tokens
- async lifecycle states: `pending` / `completed` / `failed`
- dependency relationships across async tasks

Current use:

- CTDG records `sample`, `feature_fetch`, `state_fetch`, `memory_updater`, `neighbor_attention_aggregate`, `message_generate`, `state_transition`, `state_writeback`
- DTDG records `load_snapshot`, `route_apply`, `state_fetch`, `state_transition`, `state_writeback`

### AsyncStageHandle

`AsyncStageHandle` represents a stage that can be scheduled or tracked asynchronously.

It carries:

- `token`
- `name`
- `status`
- `payload`
- `depends_on`

### StateHandle

`StateHandle` identifies the runtime state container being operated on.

Typical usage:

- CTDG: node-level memory/mailbox state
- DTDG: window-level temporal state

### StateDelta

`StateDelta` describes the state change produced by the current step.

Typical usage:

- CTDG: memory updater and message generation effects
- DTDG: snapshot propagation and temporal fusion effects

### StateWriteback

`StateWriteback` describes how a `StateDelta` is committed back into a `StateHandle`.

This allows unified inspection, logging, and checkpoint-friendly metadata without erasing the execution differences between CTDG and DTDG.

## CTDG Library Stack

CTDG components are implemented in [ctdg_kernel.py](/home/zlj/StarryUniGraph/starry_unigraph/core/ctdg_kernel.py).

### Data and state objects

- `FeatureRoutePlan`
- `StateSyncPlan`
- `CTDGPartitionBook`
- `CTDGMailboxState`
- `CTDGBatch`
- `CTDGPreparedBatch`
- `CTDGStepResult`
- `CTDGRuntimeState`
- `CTDGSamplerCore`
- `CTDGKernel`

### CTDG execution pipeline

The current CTDG kernel preserves the BTS-style stage order:

1. `sample`
2. `feature_fetch`
3. `state_fetch`
4. `memory_updater`
5. `neighbor_attention_aggregate`
6. `message_generate`
7. `state_transition`
8. `state_writeback`

This is implemented by `CTDGKernel`.

Main methods:

- `iter_batches()`
- `_state_fetch()`
- `_materialize()`
- `_memory_updater()`
- `_neighbor_attention_aggregate()`
- `_message_generate()`
- `_state_writeback()`
- `_run_pipeline()`
- `execute_train()`
- `execute_eval()`
- `execute_predict()`

### CTDG artifact model

The CTDG provider writes:

- `artifacts/{dataset}/meta/artifacts.json`
- `artifacts/{dataset}/partitions/manifest.json`
- `artifacts/{dataset}/routes/manifest.json`
- `artifacts/{dataset}/sampling/index.json`

These files encode:

- graph mode
- partition algorithm
- number of parts
- feature route plan
- state sync plan
- sampling lookup metadata

### CTDG native sampler integration

The repository now contains a vendored BTS sampler integration:

- vendored source: [vendor/bts_sampler](/home/zlj/StarryUniGraph/starry_unigraph/vendor/bts_sampler)
- packaged shared object: [libstarrygl_sampler.so](/home/zlj/StarryUniGraph/starry_unigraph/lib/libstarrygl_sampler.so)
- runtime loader: [loader.py](/home/zlj/StarryUniGraph/starry_unigraph/lib/loader.py)
- Python wrapper: [bts_sampler.py](/home/zlj/StarryUniGraph/starry_unigraph/native/bts_sampler.py)

Available native helpers:

- `is_bts_sampler_available()`
- `build_temporal_neighbor_block(...)`
- `BTSNativeSampler`

Current status:

- BTS sampler binary and source are merged into the repository.
- CTDG sampler core can attach a native sampler through `attach_native_sampler(...)`.
- The default session flow does not yet automatically drive the full CTDG kernel through the native sampler output.

So the repository now contains the native sampler asset and wrapper, but the end-to-end kernel path is still only partially migrated.

## DTDG Library Stack

DTDG components are implemented in [dtdg_kernel.py](/home/zlj/StarryUniGraph/starry_unigraph/core/dtdg_kernel.py).

### Data and state objects

- `SnapshotRoutePlan`
- `DTDGPartitionBook`
- `DTDGWindowState`
- `DTDGBatch`
- `DTDGStepResult`
- `DTDGRuntimeState`
- `DTDGSnapshotCore`
- `DTDGKernel`

### DTDG execution pipeline

The current DTDG kernel preserves the Flare-style stage order:

1. `load_snapshot`
2. `route_apply`
3. `state_fetch`
4. `state_transition`
5. `state_writeback`

This is implemented by `DTDGKernel`.

Main methods:

- `iter_batches()`
- `_apply_route()`
- `_state_fetch()`
- `_snapshot_propagation()`
- `_state_transition()`
- `_state_writeback()`
- `_run_pipeline()`
- `execute_train()`
- `execute_eval()`
- `execute_predict()`

### DTDG artifact model

The DTDG provider writes:

- `artifacts/{dataset}/meta/artifacts.json`
- `artifacts/{dataset}/partitions/manifest.json`
- `artifacts/{dataset}/routes/manifest.json`
- `artifacts/{dataset}/snapshots/index.json`

These files encode:

- graph mode
- partition algorithm
- snapshot count
- route plan
- window metadata

### DTDG high-performance backend status

Flare high-performance preprocessing and loading are not yet fully merged.

Not yet migrated:

- Flare `PartitionData`
- Flare `STGraphLoader`
- Flare `RNNStateManager`
- route/state-aware pinned-memory loading
- asynchronous remap and snapshot transfer path

This means DTDG currently has:

- unified interface
- explicit kernel stages
- artifact model

but not the original Flare high-performance data path.

## Configuration System

The configuration system is defined in [schema.py](/home/zlj/StarryUniGraph/starry_unigraph/config/schema.py) and defaults in [default.yaml](/home/zlj/StarryUniGraph/starry_unigraph/config/default.yaml).

### Main sections

- `model`
- `data`
- `train`
- `runtime`
- `sampling`
- `features`
- `graph`
- `dist`

### Graph mode dispatch

`model.family` is resolved through the model registry and mapped to:

- `ctdg`
- `dtdg`

This drives provider resolution and kernel selection.

### Validation behavior

The schema layer:

- fills defaults
- validates required config paths
- infers graph mode when possible
- warns about inactive fields
- raises hard errors for missing required fields

## Registry System

The registry layer decouples names from implementations.

### ModelRegistry

Defined in [model_registry.py](/home/zlj/StarryUniGraph/starry_unigraph/registry/model_registry.py).

Purpose:

- map `model.name` or `model.family` to a `ModelSpec`
- determine `graph_mode`

### ProviderRegistry

Defined in [provider_registry.py](/home/zlj/StarryUniGraph/starry_unigraph/registry/provider_registry.py).

Purpose:

- map `ctdg` to `CTDGProvider`
- map `dtdg` to `DTDGProvider`

### TaskRegistry

Defined in [task_registry.py](/home/zlj/StarryUniGraph/starry_unigraph/registry/task_registry.py).

Purpose:

- resolve task adapters
- separate task semantics from graph-family execution

## Runtime and Checkpointing

The runtime abstraction lives in [runtime/base.py](/home/zlj/StarryUniGraph/starry_unigraph/runtime/base.py).

The current provider stack uses a lightweight runtime adapter from [providers/common.py](/home/zlj/StarryUniGraph/starry_unigraph/providers/common.py).

Checkpoint I/O lives in [io.py](/home/zlj/StarryUniGraph/starry_unigraph/checkpoint/io.py).

Current checkpoint payload:

- `model_state`
- `optimizer_state`
- `scheduler_state`
- `runtime_state`
- `provider_meta`
- `config`
- `epoch`
- `global_step`

Current runtime state includes provider-mapped kernel state, for example:

- CTDG
  - `memory_state`
  - `mailbox_state`
  - `sampler_state`
  - `executor_state`
- DTDG
  - `window_state`
  - `snapshot_state`
  - `route_cache`
  - `executor_state`

## Preprocessing Layer

The abstract preprocessing contract is defined in [base.py](/home/zlj/StarryUniGraph/starry_unigraph/preprocess/base.py).

The provider-specific preprocessors currently perform:

- artifact directory creation
- partition/route/index manifest generation
- artifact metadata export

Current limitation:

- preprocessors do not yet perform full BTS/Flare-native preprocessing.
- CTDG event ordering and true native sampler index generation are not fully migrated.
- DTDG snapshot chunking and Flare-native `PartitionData` generation are not fully migrated.

## CLI

CLI entrypoint is [main.py](/home/zlj/StarryUniGraph/starry_unigraph/cli/main.py).

Commands:

- `prepare`
- `train`
- `predict`
- `resume`

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

- sampling-first execution
- route-aware feature fetch
- memory/mailbox update after materialization

Trace output contains:

- `pipeline`
- `stage_payloads`
- `state`

### DTDG pipeline semantics

The library preserves:

- snapshot loading
- route application
- sparse-dense operator stage
- window aggregation
- recurrent-like state storage

Trace output contains:

- `pipeline`
- `stage_payloads`
- `state`
- `spmm_output`
- `aggregated`

## What Is Already Merged From BTS and Flare

### Already merged

- Unified config/session/provider/checkpoint architecture
- BTS sampler source and binary assets
- BTS sampler Python wrapper and loader
- Unified kernel protocol
- CTDG kernel with BTS-style stage structure
- DTDG kernel with Flare-style stage structure

### Not fully merged yet

- full BTS native sampler-driven CTDG training path
- BTS memory/mailbox implementation parity
- Flare `PartitionData`
- Flare `STGraphLoader`
- Flare `RNNStateManager`
- Flare high-performance preprocessing and asynchronous data movement

## Testing

Tests are currently in [test_session.py](/home/zlj/StarryUniGraph/tests/test_session.py).

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

Current environment note:

- The repository compiles with `python -m compileall`.
- `pytest` is not installed in the current environment, so test files exist but may need local runner setup to execute through `pytest`.

## Recommended Next Steps

If the goal is full backend parity rather than only unified architecture, the next migrations should be:

1. Connect `CTDGKernel` batch generation to the vendored BTS native sampler end-to-end.
2. Migrate BTS memory/mailbox critical path into repository-owned native/runtime modules.
3. Migrate Flare `PartitionData` into `starry_unigraph`.
4. Migrate Flare `STGraphLoader` and state manager into `starry_unigraph`.
5. Replace current DTDG synthetic batch materialization with Flare-native partition/snapshot loading.

## Summary

`StarryUniGraph` is currently a unified dynamic graph library with:

- one public session API
- one config system
- one checkpoint format
- one execution protocol
- two family-specific kernels

It already contains the structural foundation needed to converge BTS-style CTDG and Flare-style DTDG under one library, while still preserving their original pipeline semantics. The remaining work is mostly backend migration and performance-path replacement, not architecture invention.
