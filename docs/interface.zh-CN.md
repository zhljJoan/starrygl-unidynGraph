# StarryUniGraph 接口文档

[English Version](./interface.md)

本文档定义 StarryUniGraph 在 CTDG/DTDG 统一训练中的核心接口与调用契约。

## 1. Session 层接口

统一入口：`SchedulerSession`。

```python
from starry_unigraph import SchedulerSession

session = SchedulerSession.from_config("configs/tgn_wiki.yaml")
session.prepare_data()
session.build_runtime()
result = session.run_task()
```

### 1.1 公共方法
- `SchedulerSession.from_config(config_or_path, dataset_path=None, overrides=None)`
- `prepare_data() -> PreparedArtifacts`
- `build_runtime() -> RuntimeBundle`
- `run_epoch(split="train") -> dict`
- `run_task() -> dict`
- `evaluate(split="test") -> dict`
- `predict(split="test") -> PredictionResult`
- `save_checkpoint(path) -> Path`
- `load_checkpoint(path) -> dict`

### 1.2 `run_epoch` / `run_task` 输出关键字段
- `split`, `steps`, `loss`, `metrics`, `elapsed_s`
- `outputs`：来自 provider 的逐 batch 输出

## 2. Provider 生命周期契约

CTDG 与 DTDG Provider 统一遵循以下钩子：

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

## 3. Artifact 契约

所有预处理器统一遵循：

- `prepare_raw(session_ctx)`
- `build_partitions(session_ctx)`
- `build_runtime_artifacts(session_ctx) -> PreparedArtifacts`

### 3.1 通用目录布局

```text
artifacts/<data.name>/
  meta/artifacts.json
  partitions/manifest.json
  routes/manifest.json
```

### 3.2 管线特定输出
- CTDG：`sampling/index.json`
- DTDG `flare_native`：`flare/manifest.json`, `flare/part_XXX.pth`
- DTDG `chunked`：`snapshots/manifest.json`, `snapshots/.../chunk_XXX.pth`, `clusters/.../cluster_manifest.json`

## 4. Runtime Bundle 与关键数据类型

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
- `is_distributed` 属性

## 5. CTDG 运行时接口

CTDG 主要由 `CTDGProvider` + `CTDGOnlineRuntime` 实现。

### 5.1 Batch 输入
`CTDGDataBatch` 字段：
- `index`, `split`, `event_ids`
- `src`, `dst`, `ts`, `edge_feat`
- 属性：`size`

### 5.2 Step 输出结构（train/eval）
典型输出：

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

### 5.3 分布式 memory 同步接口
`CTDGMemoryBank` 关键方法：
- `gather(node_ids)`
- `read_mailbox(node_ids)`
- `read_mailbox_ts(node_ids)`
- `submit_async_memory_sync(ctx, node_ids, values, timestamps)`
- `submit_async_mail_sync(ctx, node_ids, mail_slots, mail_ts)`
- `wait_pending_syncs()`

### 5.4 Historical Cache 接口
`CTDGHistoricalCache` 关键方法：
- `historical_check(node_ids, current_memory) -> mask`
- `update_cache(slot_indices, new_memory, new_ts)`
- `slot_indices(node_ids) -> Tensor`
- `bind_shared_nodes(num_nodes, shared_node_ids)`

## 6. DTDG 运行时接口

DTDG 由 `DTDGProvider` 管理，当前支持两条 pipeline：
- `flare_native`
- `chunked`

### 6.1 训练/评估/预测统一输出
DTDG step 函数统一返回以下字段：
- `loss`（若可计算）
- `predictions`
- `targets`
- `meta`

### 6.2 Flare 状态接口
在 `flare_native` 中，序列状态通过 RNN state manager 与 snapshot 流程维护。

## 7. 配置接口

两类范式共享配置段：
- `model.*`
- `data.*`
- `train.*`
- `runtime.*`
- `graph.*`
- `dist.*`

CTDG 专属配置段：
- `ctdg.pipeline`
- `ctdg.mailbox_slots`
- `ctdg.historical_alpha`
- `ctdg.async_sync`
- `ctdg.ada_param_enabled`
- `ctdg.dim_time`, `ctdg.num_head`, `ctdg.dropout`, `ctdg.att_dropout`

DTDG 专属配置段：
- `dtdg.pipeline`
- `dtdg.chunk_order`
- `dtdg.chunk_decay`
- `dtdg.num_full_snaps`

## 8. 最小可运行示例

### 8.1 CTDG 分布式
```bash
torchrun --nproc_per_node=4 train_tgn_dist.py --dataset WIKI --epochs 2
```

### 8.2 DTDG 分布式
```bash
bash run_mpnn_lstm_4gpu.sh all
```

## 9. 扩展建议

- CTDG/DTDG 共享编排契约，但 kernel 细节可独立演进。
- 新模型建议通过 model/provider/task registry 接入，并复用 provider 生命周期。
- 新分布式优化模块建议通过 `meta` 统一暴露监控指标，便于 session 层观测和回归分析。
