# StarryUniGraph 接口文档

[English Version](./interface.md)

本文档定义 StarryUniGraph 在 CTDG/DTDG 统一训练中的核心接口与调用契约。

## 1. Session 层接口

### 1.1 DTDG：`SchedulerSession`

离散时间动态图训练的主入口。

```python
from starry_unigraph import SchedulerSession

session = SchedulerSession.from_config("configs/mpnn_lstm_4gpu.yaml")
session.prepare_data()
session.build_runtime()
result = session.run_task()
```

公共方法：
- `SchedulerSession.from_config(config_or_path, dataset_path=None, overrides=None)`
- `prepare_data() -> PreparedArtifacts`
- `build_runtime() -> RuntimeBundle`
- `run_epoch(split="train") -> dict`
- `run_task() -> dict`
- `evaluate(split="test") -> dict`
- `predict(split="test") -> PredictionResult`
- `save_checkpoint(path) -> Path`
- `load_checkpoint(path) -> dict`

`run_epoch` / `run_task` 输出关键字段：
- `split`, `steps`, `loss`, `metrics`, `elapsed_s`
- `outputs`：逐 batch 的 step 函数输出

### 1.2 CTDG：`CTDGSession`

连续时间动态图训练（TGN 风格）的入口。

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

公共方法：
- `prepare_data(session_ctx) -> PreparedArtifacts`
- `build_runtime(session_ctx)`
- `iter_train(session_ctx)` / `iter_eval(session_ctx, split="val")` / `iter_predict(session_ctx, split="test")`
- `train_step(batch) -> dict` / `eval_step(batch) -> dict` / `predict_step(batch) -> dict`
- `save_checkpoint(path)` / `load_checkpoint(path)`

## 2. Artifact 契约

所有预处理器统一遵循：

- `prepare_raw(session_ctx)`
- `build_partitions(session_ctx)`
- `build_runtime_artifacts(session_ctx) -> PreparedArtifacts`

### 2.1 通用目录布局

```text
artifacts/<data.name>/
  meta/artifacts.json
  partitions/manifest.json
  routes/manifest.json
```

### 2.2 管线特定输出
- CTDG：`sampling/index.json`
- DTDG `flare_native`：`flare/manifest.json`, `flare/part_XXX.pth`

## 3. Runtime Bundle 与关键数据类型

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
- `is_distributed` 属性

### 3.4 `SessionContext`
- `config: dict`
- `artifact_root: Path`
- `dist: DistributedContext`
- `provider_state: dict`

### 3.5 `PredictionResult`
- `predictions`
- `targets`
- `meta`

## 4. CTDG 运行时接口

CTDG 在 `runtime/online/` 中实现，通过 `CTDGSession` 暴露接口。

### 4.1 Batch 输入
`CTDGDataBatch` 字段：
- `index`, `split`, `event_ids`
- `src`, `dst`, `ts`, `edge_feat`
- 属性：`size`

### 4.2 Step 输出结构（train/eval）
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

### 4.3 采样接口
`NativeTemporalSampler` 关键方法：
- `sample_from_nodes(nodes, timestamps) -> CTDGSampleOutput`
- `reset()`
- `describe() -> dict`

### 4.4 分布式 memory 同步接口
`CTDGMemoryBank` 关键方法：
- `gather(node_ids)`
- `read_mailbox(node_ids)` / `read_mailbox_ts(node_ids)`
- `write_mailbox(node_ids, mail_slots, mail_ts)`
- `submit_async_memory_sync(ctx, node_ids, values, timestamps)`
- `submit_async_mail_sync(ctx, node_ids, mail_slots, mail_ts)`
- `sync_updates()`

### 4.5 Historical Cache 接口
`CTDGHistoricalCache` 关键方法：
- `historical_check(node_ids, current_memory) -> mask`
- `update_cache(slot_indices, new_memory, new_ts)`
- `slot_indices(node_ids) -> Tensor`
- `synchronize_shared_update()`

### 4.6 Feature Route 接口
`CTDGFeatureRoute` 关键方法：
- `exchange(features) -> routed_features`
- `describe() -> dict`

### 4.7 工厂函数
`build_ctdg_runtime(session_ctx) -> (CTDGOnlineRuntime, RuntimeBundle)` — 一次调用构建 dataset、sampler、memory、model、optimizer 和 DDP 包装。

## 5. DTDG 运行时接口

DTDG 在 `runtime/flare/` 中实现，由 `SchedulerSession` 直接编排。

### 5.1 数据结构

`PartitionData` 关键接口：
- `__len__()` — 快照数量
- `__getitem__(idx)` — 切片快照
- `to(device) -> PartitionData` / `pin_memory() -> PartitionData`
- `to_blocks() -> list[DGLBlock]`
- `save(path)` / `load(path)`
- `num_snaps` / `num_dst_nodes` 属性

`TensorData` / `RouteData` — 压缩张量存储与每快照路由描述符。

### 5.2 FlareRuntimeLoader 接口
`FlareRuntimeLoader`（位于 `runtime/flare/session_loader.py`）关键方法：
- `from_partition_data(data, device, rank, world_size, config) -> FlareRuntimeLoader`（类方法）
- `iter_train(split="train") -> Iterator[STGraphBlob]`
- `iter_eval(split="val") -> Iterator[DTDGBatch]`
- `iter_predict(split="test") -> Iterator[DTDGBatch]`
- `dump_state() -> dict`
- `describe_window_state() -> dict` / `describe_route_cache() -> dict`

### 5.3 STGraphLoader 接口
`STGraphLoader` 关键方法：
- `from_partition_data(data, device, chunk_index, rank, size) -> STGraphLoader`（类方法）
- `fetch_graph(idx) -> DGLBlock` — 通过 CUDA stream 异步获取
- `build_snapshot_index() -> dict`

### 5.4 RNN 状态管理接口
`RNNStateManager` 关键方法：
- `add(graph)` — 将图加入滑动窗口；为图 patch `flare_fetch_state()` / `flare_store_state()` 方法
- `state_detach()` — 将所有存储状态从计算图中分离

`STGraphBlob` 关键接口：
- `current_graph` 属性 — 窗口中最后一个图
- `snapshot_index` 属性
- `__iter__()` — 遍历窗口中的图

### 5.5 Route 接口（Flare）
`Route` 关键接口：
- `forward(features) -> Tensor` — 可微分 all-to-all 交换
- `send_len` / `recv_len` 属性

### 5.6 训练/评估/预测契约

训练步输入：`STGraphBlob`（多帧）
评估/预测步输入：`DTDGBatch`（单帧）

关键训练函数（来自 `runtime/flare/training.py`）：
- `init_flare_training(runtime, session_ctx, partition_data, device)` — 构建模型 + 优化器 + DDP
- `run_flare_train_step(runtime, batch, kernel_output) -> dict`
- `run_flare_eval_step(runtime, batch, kernel_output) -> dict`
- `run_flare_predict_step(runtime, batch, kernel_output) -> dict`

### 5.7 DTDG 数据类型（位于 `runtime/flare/session_loader.py`）
- `DTDGBatch` — 单帧评估/预测 batch（index, split, window_size, route_plan, graph, graph_meta）
- `DTDGWindowState` — 窗口追踪（window_size, last_snapshot, stored_windows）
- `SnapshotRoutePlan` — route_type, cache_policy

### 5.8 模型接口（Flare）

所有 Flare 模型实现 `forward(graphs) -> Tensor`：

- `FlareEvolveGCN(in_dim, hidden_dim, out_dim)` — EvolveGCN-H
- `FlareTGCN(in_dim, hidden_dim, out_dim)` — Temporal GCN
- `FlareMPNNLSTM(in_dim, hidden_dim, out_dim)` — MPNN-LSTM
- `GCNStack(in_dim, hidden_dim, out_dim, num_layers)` — 多层 GCN

工厂函数：`build_flare_model(name, in_dim, hidden_dim, out_dim)`

## 6. 预处理接口

### 6.1 DTDG 预处理
`FlareDTDGPreprocessor`（位于 `preprocess/dtdg.py`）：
- 继承 `GraphPreprocessor`：`prepare_raw()` → `build_partitions()` → `build_runtime_artifacts()`
- 通过 METIS 分区写出 `flare/part_NNN.pth` 文件

artifact 校验工具（同在 `preprocess/dtdg.py`）：
- `validate_artifacts(prepared, expected_graph_mode, expected_num_parts)`
- `load_prepared_from_disk(artifact_root) -> PreparedArtifacts`

### 6.2 CTDG 预处理
`CTDGPreprocessor`（位于 `preprocess/ctdg.py`）：
- 继承 `GraphPreprocessor`：`prepare_raw()` → `build_partitions()` → `build_runtime_artifacts()`
- 写出分区 manifest + feature route plan

## 7. 数据切分工具

`runtime/_split.py` 提供 flare 和 online 两侧共用的切分逻辑：
- `normalize_split_ratio(split_ratio) -> dict` — 确保 train+val+test 之和为 1.0
- `split_bounds(total, split, split_ratio) -> (int, int)` — 返回 `(start, end)` 索引区间

## 8. 配置接口

两类范式共享配置段：
- `model.*`, `data.*`, `train.*`, `runtime.*`, `graph.*`, `dist.*`

CTDG 专属：
- `ctdg.mailbox_slots`, `ctdg.historical_alpha`, `ctdg.async_sync`, `ctdg.ada_param_enabled`
- `ctdg.dim_time`, `ctdg.num_head`, `ctdg.dropout`, `ctdg.att_dropout`

DTDG 专属：
- `dtdg.chunk_order`, `dtdg.chunk_decay`, `dtdg.num_full_snaps`

## 9. 最小可运行示例

### 9.1 CTDG 分布式
```bash
torchrun --nproc_per_node=4 train_tgn_dist.py --dataset WIKI --epochs 2
```

### 9.2 DTDG 分布式
```bash
python train_mpnn_lstm_4gpu.py --mode prepare
torchrun --nproc_per_node=4 train_mpnn_lstm_4gpu.py --mode train
```

### 9.3 DTDG 逐步（Python）
```python
from starry_unigraph import SchedulerSession

# 直接使用 SchedulerSession，无 provider 层
session = SchedulerSession(session_ctx=ctx, model_spec=model_spec, task_adapter=task)
session.prepare_data()
session.build_runtime()

for epoch in range(epochs):
    train_result = session.run_epoch(split="train")
    eval_result  = session.run_epoch(split="val")

session.save_checkpoint("checkpoint.pkl")
```

### 9.4 CTDG 逐步（Python）
```python
from starry_unigraph.runtime.online import CTDGSession

session = CTDGSession()
session.prepare_data(ctx)
session.build_runtime(ctx)
for batch in session.iter_train(ctx):
    result = session.train_step(batch)
```

## 10. 扩展建议

- CTDG/DTDG 共享 artifact 布局和 checkpoint 约定，但运行时细节可独立演进。
- 新 DTDG 模型通过 `ModelRegistry` 注册，通过 `build_flare_model` 实例化。
- 新分布式优化模块建议通过 `meta` 统一暴露监控指标，便于 session 层观测。
- `backends/` 包提供从 `runtime/` 子包的向后兼容重导出。
