# StarryUniGraph Interface Documentation

[中文版本](./interface.zh-CN.md)

This document defines the practical interfaces used by StarryUniGraph for unified CTDG/DTDG training.

## 1. Session-Level API

### 1.1 DTDG: `SchedulerSession`

Primary entry for Discrete-Time Dynamic Graph training.

```python
from starry_unigraph import SchedulerSession

session = SchedulerSession.from_config("configs/mpnn_lstm_4gpu.yaml")
session.prepare_data()
session.build_runtime()
result = session.run_task()
```

Public methods:
- `SchedulerSession.from_config(config_or_path, dataset_path=None, overrides=None)`
- `prepare_data() -> PreparedArtifacts`
- `build_runtime() -> RuntimeBundle`
- `run_epoch(split="train") -> dict`
- `run_task() -> dict`
- `evaluate(split="test") -> dict`
- `predict(split="test") -> PredictionResult`
- `save_checkpoint(path) -> Path`
- `load_checkpoint(path) -> dict`

`run_epoch` / `run_task` output key fields:
- `split`, `steps`, `loss`, `metrics`, `elapsed_s`
- `outputs`: per-batch payloads from step functions

### 1.2 CTDG: `CTDGSession`

Entry for Continuous-Time Dynamic Graph training (TGN-style).

```python
from starry_unigraph.runtime.online import CTDGSession
from starry_unigraph.types import SessionContext

session = CTDGSession()
session.prepare_data(ctx)
session.build_runtime(ctx)
for batch in session.iter_train(ctx):
    result = session.train_step(batch)
session.save_checkpoint("ckpt.pt")
```

Public methods:
- `prepare_data(session_ctx) -> PreparedArtifacts`
- `build_runtime(session_ctx)`
- `iter_train(session_ctx)` / `iter_eval(session_ctx, split="val")` / `iter_predict(session_ctx, split="test")`
- `train_step(batch) -> dict` / `eval_step(batch) -> dict` / `predict_step(batch) -> dict`
- `save_checkpoint(path)` / `load_checkpoint(path)`

## 2. Artifact Contract

All preprocessors follow:

- `prepare_raw(session_ctx)`
- `build_partitions(session_ctx)`
- `build_runtime_artifacts(session_ctx) -> PreparedArtifacts`

### 2.1 Common artifact layout

```text
artifacts/<data.name>/
  meta/artifacts.json
  partitions/manifest.json
  routes/manifest.json
```

### 2.2 Pipeline-specific outputs
- CTDG: `sampling/index.json`
- DTDG `flare_native`: `flare/manifest.json`, `flare/part_XXX.pth`

## 3. Runtime Bundle and Data Types

### 3.1 `RuntimeBundle`
- `model`
- `optimizer`
- `scheduler`
- `state: dict[str, Any]`

### 3.2 `PreparedArtifacts`
- `meta_path: Path`
- `directories: dict[str, Path]`
- `provider_meta: dict[str, Any]`

### 3.3 `DistributedContext`
- `backend`, `world_size`, `rank`, `local_rank`, `local_world_size`
- `master_addr`, `master_port`, `init_method`, `launcher`
- `is_distributed` property

### 3.4 `SessionContext`
- `config: dict`
- `artifact_root: Path`
- `dist: DistributedContext`
- `provider_state: dict`

### 3.5 `PredictionResult`
- `predictions`
- `targets`
- `meta`

## 4. CTDG Runtime Interface

CTDG is implemented in `runtime/online/` and exposed via `CTDGSession`.

### 4.1 Batch input
`CTDGDataBatch` fields:
- `index`, `split`, `event_ids`
- `src`, `dst`, `ts`, `edge_feat`
- property: `size`

### 4.2 Step output schema (train/eval)
```python
{
  "loss": float,
  "predictions": list[float],
  "targets": list[int],
  "meta": {
    "split": str,
    "batch_size": int,
    "sample": dict,
    "metrics": {"ap": float, "auc": float, "mrr": float},
    "memory": dict,
    "sync_wait_ms": float,
    "sync_submit_ms": float,
    "step_ms": float,
  }
}
```

### 4.3 Sampling interface
`NativeTemporalSampler` key methods:
- `sample_from_nodes(nodes, timestamps) -> CTDGSampleOutput`
- `reset()`
- `describe() -> dict`

### 4.4 Distributed memory sync interfaces
`CTDGMemoryBank` key methods:
- `gather(node_ids)`
- `read_mailbox(node_ids)` / `read_mailbox_ts(node_ids)`
- `write_mailbox(node_ids, mail_slots, mail_ts)`
- `submit_async_memory_sync(ctx, node_ids, values, timestamps)`
- `submit_async_mail_sync(ctx, node_ids, mail_slots, mail_ts)`
- `sync_updates()`

### 4.5 Historical cache interface
`CTDGHistoricalCache` key methods:
- `historical_check(node_ids, current_memory) -> mask`
- `update_cache(slot_indices, new_memory, new_ts)`
- `slot_indices(node_ids) -> Tensor`
- `synchronize_shared_update()`

### 4.6 Feature route interface
`CTDGFeatureRoute` key methods:
- `exchange(features) -> routed_features`
- `describe() -> dict`

### 4.7 Factory
`build_ctdg_runtime(session_ctx) -> (CTDGOnlineRuntime, RuntimeBundle)` — builds dataset, sampler, memory, model, optimizer, DDP wrapping in one call.

## 5. DTDG Runtime Interface

DTDG is implemented in `runtime/flare/` and orchestrated directly by `SchedulerSession`.

### 5.1 Data structures

`PartitionData` key interface:
- `__len__()` — number of snapshots
- `__getitem__(idx)` — slice snapshots
- `to(device) -> PartitionData` / `pin_memory() -> PartitionData`
- `to_blocks() -> list[DGLBlock]`
- `save(path)` / `load(path)`
- `num_snaps` / `num_dst_nodes` properties

`TensorData` / `RouteData` — packed tensor storage and per-snapshot routing descriptors.

### 5.2 FlareRuntimeLoader interface
`FlareRuntimeLoader` (in `runtime/flare/session_loader.py`) key methods:
- `from_partition_data(data, device, rank, world_size, config) -> FlareRuntimeLoader` (classmethod)
- `iter_train(split="train") -> Iterator[STGraphBlob]`
- `iter_eval(split="val") -> Iterator[DTDGBatch]`
- `iter_predict(split="test") -> Iterator[DTDGBatch]`
- `dump_state() -> dict`
- `describe_window_state() -> dict` / `describe_route_cache() -> dict`

### 5.3 STGraphLoader interface
`STGraphLoader` key methods:
- `from_partition_data(data, device, chunk_index, rank, size) -> STGraphLoader` (classmethod)
- `fetch_graph(idx) -> DGLBlock` — async fetch via CUDA stream
- `build_snapshot_index() -> dict`

### 5.4 RNN state management interface
`RNNStateManager` key methods:
- `add(graph)` — adds graph to sliding window; patches with `flare_fetch_state()` / `flare_store_state()`
- `state_detach()` — detaches all stored states

`STGraphBlob` key interface:
- `current_graph` property — last graph in window
- `snapshot_index` property
- `__iter__()` — iterate over graphs in window

### 5.5 Route interface (Flare)
`Route` key interface:
- `forward(features) -> Tensor` — differentiable all-to-all exchange
- `send_len` / `recv_len` properties

### 5.6 Training/eval/predict contract

Training step input: `STGraphBlob` (multi-frame)
Eval/predict step input: `DTDGBatch` (single frame)

Key training functions (from `runtime/flare/training.py`):
- `init_flare_training(runtime, session_ctx, partition_data, device)` — builds model + optimizer + DDP
- `run_flare_train_step(runtime, batch, kernel_output) -> dict`
- `run_flare_eval_step(runtime, batch, kernel_output) -> dict`
- `run_flare_predict_step(runtime, batch, kernel_output) -> dict`

### 5.7 DTDG data types (in `runtime/flare/session_loader.py`)
- `DTDGBatch` — single-frame eval/predict batch (index, split, window_size, route_plan, graph, graph_meta)
- `DTDGWindowState` — window tracking (window_size, last_snapshot, stored_windows)
- `SnapshotRoutePlan` — route_type, cache_policy

### 5.8 Model interface (Flare)

All Flare models implement `forward(graphs) -> Tensor`:

- `FlareEvolveGCN(in_dim, hidden_dim, out_dim)` — EvolveGCN-H
- `FlareTGCN(in_dim, hidden_dim, out_dim)` — Temporal GCN
- `FlareMPNNLSTM(in_dim, hidden_dim, out_dim)` — MPNN-LSTM
- `GCNStack(in_dim, hidden_dim, out_dim, num_layers)` — multi-layer GCN

Factory: `build_flare_model(name, in_dim, hidden_dim, out_dim)`

## 6. Preprocessing Interface

### 6.1 DTDG preprocessing
`FlareDTDGPreprocessor` (in `preprocess/dtdg.py`):
- Inherits `GraphPreprocessor`: `prepare_raw()` → `build_partitions()` → `build_runtime_artifacts()`
- Writes `flare/part_NNN.pth` files via METIS partitioning

Artifact validation utilities (also in `preprocess/dtdg.py`):
- `validate_artifacts(prepared, expected_graph_mode, expected_num_parts)`
- `load_prepared_from_disk(artifact_root) -> PreparedArtifacts`

### 6.2 CTDG preprocessing
`CTDGPreprocessor` (in `preprocess/ctdg.py`):
- Inherits `GraphPreprocessor`: `prepare_raw()` → `build_partitions()` → `build_runtime_artifacts()`
- Writes partition manifest + feature route plan

## 7. Data Split Utilities

`runtime/_split.py` provides shared split logic used by both flare and online runtimes:
- `normalize_split_ratio(split_ratio) -> dict` — ensures train+val+test sum to 1.0
- `split_bounds(total, split, split_ratio) -> (int, int)` — returns `(start, end)` index range

## 8. Configuration Interface

Common sections:
- `model.*`, `data.*`, `train.*`, `runtime.*`, `graph.*`, `dist.*`

CTDG-specific:
- `ctdg.mailbox_slots`, `ctdg.historical_alpha`, `ctdg.async_sync`, `ctdg.ada_param_enabled`
- `ctdg.dim_time`, `ctdg.num_head`, `ctdg.dropout`, `ctdg.att_dropout`

DTDG-specific:
- `dtdg.chunk_order`, `dtdg.chunk_decay`, `dtdg.num_full_snaps`

## 9. End-to-End Examples

### 9.1 CTDG distributed
```bash
torchrun --nproc_per_node=4 train_tgn_dist.py --dataset WIKI --epochs 2
```

### 9.2 DTDG distributed
```bash
python train_mpnn_lstm_4gpu.py --mode prepare
torchrun --nproc_per_node=4 train_mpnn_lstm_4gpu.py --mode train
```

### 9.3 DTDG step-by-step (Python)
```python
from starry_unigraph import SchedulerSession
from starry_unigraph.types import SessionContext

# Use SchedulerSession directly — no provider layer
session = SchedulerSession(session_ctx=ctx, model_spec=model_spec, task_adapter=task)
session.prepare_data()
session.build_runtime()

for epoch in range(epochs):
    train_result = session.run_epoch(split="train")
    eval_result  = session.run_epoch(split="val")

session.save_checkpoint("checkpoint.pkl")
```

### 9.4 CTDG step-by-step (Python)
```python
from starry_unigraph.runtime.online import CTDGSession

session = CTDGSession()
session.prepare_data(ctx)
session.build_runtime(ctx)
for batch in session.iter_train(ctx):
    result = session.train_step(batch)
```

## 10. Extension Notes

- CTDG and DTDG share artifact layout and checkpoint conventions but keep independent runtime internals.
- New DTDG models register into `ModelRegistry` and are built via `build_flare_model`.
- New distributed optimization modules should emit metrics through `meta` to keep session-level observability consistent.
- `backends/` package provides backward-compatible re-exports from `runtime/` subpackages.
