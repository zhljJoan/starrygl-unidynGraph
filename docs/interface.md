# StarryUniGraph Interface Documentation

[中文版本](./interface.zh-CN.md)

This document defines the practical interfaces used by StarryUniGraph for unified CTDG/DTDG training.

## 1. Session-Level API

Primary entry: `SchedulerSession`.

```python
from starry_unigraph import SchedulerSession

session = SchedulerSession.from_config("configs/tgn_wiki.yaml")
session.prepare_data()
session.build_runtime()
result = session.run_task()
```

### 1.1 Public methods
- `SchedulerSession.from_config(config_or_path, dataset_path=None, overrides=None)`
- `prepare_data() -> PreparedArtifacts`
- `build_runtime() -> RuntimeBundle`
- `run_epoch(split="train") -> dict`
- `run_task() -> dict`
- `evaluate(split="test") -> dict`
- `predict(split="test") -> PredictionResult`
- `save_checkpoint(path) -> Path`
- `load_checkpoint(path) -> dict`

### 1.2 `run_epoch` / `run_task` output (key fields)
- `split`, `steps`, `loss`, `metrics`, `elapsed_s`
- `outputs`: per-batch payloads from provider step functions

## 2. Provider Lifecycle Contract

Both CTDG and DTDG providers follow the same lifecycle hooks:

- `prepare_data(session_ctx) -> PreparedArtifacts`
- `build_runtime(session_ctx) -> RuntimeBundle`
- `build_train_iterator(session_ctx, split="train") -> Iterable[Any]`
- `build_eval_iterator(session_ctx, split="val") -> Iterable[Any]`
- `build_predict_iterator(session_ctx, split="test") -> Iterable[Any]`
- `run_train_step(batch) -> dict`
- `run_eval_step(batch) -> dict`
- `run_predict_step(batch) -> dict`
- `save_checkpoint(path)`
- `load_checkpoint(path)`

## 3. Artifact Contract

All preprocessors follow:

- `prepare_raw(session_ctx)`
- `build_partitions(session_ctx)`
- `build_runtime_artifacts(session_ctx) -> PreparedArtifacts`

### 3.1 Common artifact layout

```text
artifacts/<data.name>/
  meta/artifacts.json
  partitions/manifest.json
  routes/manifest.json
```

### 3.2 Pipeline-specific outputs
- CTDG: `sampling/index.json`
- DTDG `flare_native`: `flare/manifest.json`, `flare/part_XXX.pth`
- DTDG `chunked`: `snapshots/manifest.json`, `snapshots/.../chunk_XXX.pth`, `clusters/.../cluster_manifest.json`

## 4. Runtime Bundle and Data Types

### 4.1 `RuntimeBundle`
- `model`
- `optimizer`
- `scheduler`
- `state: dict[str, Any]`

### 4.2 `PreparedArtifacts`
- `meta_path: Path`
- `directories: dict[str, Path]`
- `provider_meta: dict[str, Any]`

### 4.3 `DistributedContext`
- `backend`, `world_size`, `rank`, `local_rank`, `local_world_size`
- `master_addr`, `master_port`, `init_method`, `launcher`
- `is_distributed` property

## 5. CTDG Runtime Interface

CTDG is implemented around `CTDGProvider` + `CTDGOnlineRuntime`.

### 5.1 Batch input
`CTDGDataBatch` fields:
- `index`, `split`, `event_ids`
- `src`, `dst`, `ts`, `edge_feat`
- property: `size`

### 5.2 Step output schema (train/eval)
Typical output:

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

### 5.3 Distributed memory sync interfaces
`CTDGMemoryBank` key methods:
- `gather(node_ids)`
- `read_mailbox(node_ids)`
- `read_mailbox_ts(node_ids)`
- `submit_async_memory_sync(ctx, node_ids, values, timestamps)`
- `submit_async_mail_sync(ctx, node_ids, mail_slots, mail_ts)`
- `wait_pending_syncs()`

### 5.4 Historical cache interface
`CTDGHistoricalCache` key methods:
- `historical_check(node_ids, current_memory) -> mask`
- `update_cache(slot_indices, new_memory, new_ts)`
- `slot_indices(node_ids) -> Tensor`
- `bind_shared_nodes(num_nodes, shared_node_ids)`

## 6. DTDG Runtime Interface

DTDG is implemented around `DTDGProvider` with two pipelines:
- `flare_native`
- `chunked`

### 6.1 Training/eval/predict contract
DTDG step functions return unified payload fields:
- `loss` (if available)
- `predictions`
- `targets`
- `meta`

### 6.2 Flare-specific state handling
For `flare_native`, stateful sequence handling is managed through RNN state manager and snapshot progression.

## 7. Configuration Interface

Common sections used by both paradigms:
- `model.*`
- `data.*`
- `train.*`
- `runtime.*`
- `graph.*`
- `dist.*`

CTDG-specific section:
- `ctdg.pipeline`
- `ctdg.mailbox_slots`
- `ctdg.historical_alpha`
- `ctdg.async_sync`
- `ctdg.ada_param_enabled`
- `ctdg.dim_time`, `ctdg.num_head`, `ctdg.dropout`, `ctdg.att_dropout`

DTDG-specific section:
- `dtdg.pipeline`
- `dtdg.chunk_order`
- `dtdg.chunk_decay`
- `dtdg.num_full_snaps`

## 8. Minimal End-to-End Examples

### 8.1 CTDG distributed
```bash
torchrun --nproc_per_node=4 train_tgn_dist.py --dataset WIKI --epochs 2
```

### 8.2 DTDG distributed
```bash
bash run_mpnn_lstm_4gpu.sh all
```

## 9. Compatibility and Extension Notes

- CTDG and DTDG share orchestration contracts but keep kernel-specific internals.
- New model families should register into model/provider/task registries and follow the same provider lifecycle.
- New distributed optimization modules should emit metrics through `meta` to keep session-level observability consistent.
