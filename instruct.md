# Chunk Backend Architecture

本文档描述当前 `backends/chunk` 链路的实现契约：prepare 产物、runtime 装载、task batch 构造、CTDG sampled-fetch、DTDG propagation route、通信与模型 head 的数据约定。

## 运行模式

`starry_unigraph.backends.chunk` 当前承载两类 runtime：

- CTDG sampled edge prediction：从 canonical event store 取本 rank 负责的 event，在线负采样，调用 MemShare-style native temporal sampler，再按 sampled read set 构建动态 fetch plan。
- DTDG snapshot/chunk execution：从 `PartitionData` snapshot 路径 materialize CSC-backed DGL block，模型层使用预构建 `ChunkPropagationRoute` 做 activation all-to-all。

核心边界：

- `events/canonical_events.pth` 是全局 event batch 权威表。
- `sampling/temporal_index_part_{rank:03d}.pth` 是 rank-local temporal neighbor sampling 索引。
- CTDG 主链路不预构建 `SpatialRouteData`；remote feature/memory 读取来自 sampled packed `read_dist_index`。
- DTDG 的 activation 通信走 `starry_unigraph.models.layers.route.ChunkPropagationRoute`，必须在模型层依赖 remote activation 的位置调用。

## 目录与核心文件

```text
starry_unigraph/preprocess/chunk.py
  ChunkPreprocessor: raw -> prepare artifacts 的顶层入口

starry_unigraph/backends/chunk/prepare/
  pipeline.py              placement/chunk/load/rebalance/memory route 组合入口
  chunk_assignment.py      node_to_chunk 与 chunk CSR
  load_stats.py            chunk load 统计
  rebalancer.py            chunk owner rebalance
  route_builder.py         MemoryRouteData / legacy SpatialRouteData 构建
  propagation_builder.py   DTDG ChunkPropagationRoute 预构建
  time_slice.py            time_ptr 构建工具

starry_unigraph/backends/chunk/data/
  partition.py             PartitionData per-rank graph container
  graph_store.py           canonical events/targets + CTDG/DTDG lightweight views
  plans.py                 ExecutionUnit / FetchPlan / PropagationPlan / StateSyncPlan
  route.py                 SpatialRouteData / MemoryRouteData / CPUMemoryLayout
  dist_index.py            packed DistIndex 编解码
  comm.py                  CommPipeline handle 化通信
  batch.py                 BatchData

starry_unigraph/backends/chunk/runtime/
  loader.py                ChunkRuntimeLoader artifact loader + train/eval step glue
  event_engine.py          MemShareEventEngine native temporal sampling
  sampler.py               NegativeSamplerHook / EdgePredictNegativeSampler / MFGBuilder
  task_adapter.py          edge/node classify/regress task adapters
  train_step.py            run_batch
  weight_version_manager.py ordered async grad sync helper

starry_unigraph/models/
  layers/route.py          ChunkPropagationRoute autograd all-to-all
  task_head.py             id_map-aware edge/node heads
  id_map.py                compact row lookup helper
```

## Prepare 总流程

顶层入口：

- `ChunkPreprocessor.prepare_raw()` in `starry_unigraph/preprocess/chunk.py`
- `ChunkPreprocessor.build_partitions()` in `starry_unigraph/preprocess/chunk.py`
- `ChunkPreprocessor.build_runtime_artifacts()` in `starry_unigraph/preprocess/chunk.py`
- `run_chunk_preprocess_from_config(config)` in `starry_unigraph/preprocess/chunk.py`

算法流程：

```text
raw temporal events
  -> build_snapshot_dataset_from_events()
  -> time_ptr
  -> prepare_chunks()
     -> build_node_partition()
     -> build chunk assignment
     -> compute chunk load by slice
     -> rebalance chunk owners
     -> build memory routes
  -> build event_owner / target_owner
  -> build placement + packed DistIndex
  -> write canonical events / targets / chunks / indices
  -> build PartitionData per rank
  -> build TemporalIndexView per rank
  -> build DTDG propagation routes
  -> write manifests
```

当前 `ChunkPreprocessor.build_partitions()` 使用 `_time_ptr_from_snapshots(raw_dataset)` 生成 `time_ptr`。`prepare/time_slice.py` 提供 CTDG adaptive/fixed-window 和 DTDG snapshot 的工具函数，供后续策略接入。

### PrepareArtifacts

定义：`PrepareArtifacts` in `starry_unigraph/backends/chunk/prepare/pipeline.py`

```text
assignment              ChunkAssignment
node_owner              Tensor[num_nodes]
node_partition          Tensor[num_nodes]  # master/initial partition
node_to_partition       Tensor[num_nodes]
hot_node_mask           bool[num_nodes]
hot_node_ids            int64[num_hot]
replica_mask            bool[num_nodes]
chunk_load_by_slice     Optional[float32[T, C]]
mem_routes              Optional[list[list[MemoryRouteData]]]
spatial_routes          Optional[list[list[SpatialRouteData]]] legacy only
time_ptr                Optional[int64[T+1]]
```

## Prepare 核心函数

### `prepare()`

文件：`starry_unigraph/backends/chunk/prepare/pipeline.py`

输入：

```text
edge_src, edge_dst              Tensor[E]
edge_timestamps                 Optional[Tensor[E]]
time_ptr                        Optional[int64[T+1]]
num_partitions                  int
num_nodes                       Optional[int]
partition_strategy              "metis" | "mem_share" | "chunk_metis_balance"
hot_topk / hot_ratio            hot node 选择参数
num_chunks_per_partition        int
build_mem_routes                bool
num_candidates                  int
```

输出：`PrepareArtifacts`

核心逻辑：

1. 选择/构造 `node_partition` 和 hot node mask。
2. 构造 `ChunkAssignment`，`node_to_chunk` 表示每个节点唯一 master chunk。
3. 若有 `time_ptr`，按 slice 计算 `chunk_load_by_slice`。
4. 调用 `rebalance_chunks()` 更新 `chunk_to_owner_partition` 并派生 `node_owner`。
5. 可选构建 `MemoryRouteData`。

### `build_node_partition()`

文件：`starry_unigraph/backends/chunk/prepare/pipeline.py`

输入：`edge_src/edge_dst/hot_edge_src/hot_edge_dst/num_partitions/strategy/hot_topk/hot_ratio`

输出：

```text
node_partition   int64[num_nodes]
hot_node_mask    bool[num_nodes]
replica_mask     bool[num_nodes]
```

算法：

- `metis`：对 cold nodes 尝试 pymetis；失败时按 degree 做 deterministic balanced fallback。
- `mem_share`：先按 hot-neighbor affinity 给 cold nodes 倾斜分配，再 fallback。
- hot nodes 的 master 由 `_assign_hot_masters()` 基于 cold-neighbor affinity 和负载惩罚选择。

### `ChunkAssignment`

文件：`starry_unigraph/backends/chunk/prepare/chunk_assignment.py`

数据结构：

```text
node_to_chunk                 int64[num_nodes]
chunk_ptr                     int64[num_chunks + 1]
chunk_nodes                   int64[num_nodes]
chunk_to_initial_partition    int64[num_chunks]
chunk_to_owner_partition      int64[num_chunks]
num_chunks_per_partition      int
```

`chunk_ptr/chunk_nodes` 是 chunk -> nodes 的 CSR 表示。访问一个 chunk 的节点为：

```text
chunk_nodes[chunk_ptr[c] : chunk_ptr[c+1]]
```

### `compute_chunk_load_by_slice()`

文件：`starry_unigraph/backends/chunk/prepare/load_stats.py`

输入：

```text
window_edge_srcs              list[Tensor]
window_edge_dsts              list[Tensor]
node_to_chunk                 Tensor[num_nodes]
chunk_to_owner_partition      Tensor[num_chunks]
node_to_partition             Tensor[num_nodes]
```

输出：`load: float32[num_slices, num_chunks]`

算法：

- 每个 slice 内用 `node_to_chunk[edge_dst]` 得到 dst chunk。
- `edge_count` 和 `remote_edge_count` 用 `scatter_add_`。
- active/remote node count 通过 encoded pair + `torch.unique` 统计。
- slice 级 Python loop 可接受；不做 edge 级 Python loop。

### `rebalance_chunks()` / `greedy_rebalance_by_slice()`

文件：`starry_unigraph/backends/chunk/prepare/rebalancer.py`

输入：`ChunkAssignment`、load stats 或 `chunk_load_by_slice`、`num_partitions`

输出：

```text
updated_assignment
node_owner = chunk_to_owner_partition[node_to_chunk]
ChunkReassignmentManifest
```

算法：

- 若传入 `chunk_load_by_slice`，维护 `[num_partitions, num_slices]` 负载矩阵。
- 按 chunk 总负载降序尝试迁移。
- 每次模拟迁移到候选 partition，选择能降低 worst per-slice peak 的目标。

### `build_memory_route_phase1()` / `assign_memory_route_ptrs()`

文件：`starry_unigraph/backends/chunk/prepare/route_builder.py`

Phase 1 输入：

```text
edge_src, edge_dst, edge_ts   Tensor[E]
time_ptr                      int64[T+1]
num_candidates                int
replica_mask                  Optional[bool[num_nodes]]
```

Phase 1 输出：`list[MemoryRouteData]`，每个 slice 一个，包含：

```text
unique_nodes   int64[D]
cand_pos       int64[D, K]   # cand_pos[:,0] 是该 slice 内最新 event/message 位置
replica_idx    Optional[int64[R]]
```

Phase 2 输入：Phase 1 routes、`node_owner`、`num_parts`、可选 `master_dist_index`

Phase 2 输出：`per_part_routes[rank][time_slice] -> MemoryRouteData`

字段：

```text
send_ptr        int64[P+1]
recv_ptr        int64[P+1]
recv_node_ids   int64[R]
unique_index    Optional[int64[D]] packed master DistIndex
recv_index      Optional[int64[R]] packed master DistIndex
```

### `build_propagation_routes()`

文件：`starry_unigraph/backends/chunk/prepare/propagation_builder.py`

输入：

```text
edge_src, edge_dst      Tensor[E]
time_ptr                int64[T+1]
node_owner              Tensor[num_nodes]
num_parts               int
num_layers              int
```

输出：

```text
routes[rank][snapshot_id][layer_id] -> ChunkPropagationRoute
```

当前 row convention：

- 每个 rank 的 local activation rows 是 `owned_nodes = where(node_owner == rank)` 的 sorted global node ids。
- `send_index` 索引 sender rank 的 owned-node activation rows。
- `recv_sizes` 表示 receiver 需要追加的 remote activation 行数。
- route 用于 DTDG 模型层 activation exchange，不用于 CTDG feature fetch。

## Prepare 输出 Artifact

默认 prepare root：

```text
meta.json
artifact_manifest.json
placement.pth

events/canonical_events.pth
targets/node_targets.pth
targets/target_pool_rank_{rank:03d}.pth

chunks/chunk_assignment.pth

indices/master_dist_index.pth
indices/read_dist_index_{rank:03d}.pth

partitions/manifest.json
partitions/part_{rank:03d}.pth

sampling/manifest.json
sampling/temporal_index_part_{rank:03d}.pth

routes/manifest.json
routes/propagation_routes_{rank:03d}.pth

mem_routes_{rank:03d}.pth
clusters/part_{rank:03d}/cluster_manifest.json
```

Legacy `spatial_routes_{rank}.pth` 默认不写。设置 `chunk.emit_legacy_spatial_routes=true` 才写；CTDG runtime 默认不加载它。

### `events/canonical_events.pth`

写入函数：`_write_canonical_artifacts()` in `starry_unigraph/preprocess/chunk.py`

内容：

```text
format            "chunk_canonical_events_v1"
schema_version    int
src               int64[E]
dst               int64[E]
ts                timestamp[E]
eid               int64[E]
event_owner       int64[E]
event_dst_chunk   int64[E]
event_split       uint8[E]     # 0=train, 1=val, 2=test
time_ptr          int64[T+1]
```

语义：

- 全局 event 列式表。
- `time_ptr[t]:time_ptr[t+1]` 是 slice `t` 的 event 区间。
- `event_owner` 决定 edge predict 正样本在哪个 rank 计算 loss。

event owner policy：

```text
if src_hot xor dst_hot:
    owner(non_hot_node)
elif both cold:
    owner(dst)
else:
    master(dst)
```

### `sampling/temporal_index_part_{rank:03d}.pth`

写入位置：`ChunkPreprocessor.build_runtime_artifacts()` in `starry_unigraph/preprocess/chunk.py`

来源：`ChunkGraphStore.temporal_index_view()` in `starry_unigraph/backends/chunk/data/graph_store.py`

内容：

```text
format       "chunk_temporal_index_v1"
indptr       int64[num_nodes + 1]
indices      int64[num_edges_local]
edge_ids     int64[num_edges_local]
timestamps   Optional[timestamp[num_edges_local]]
num_nodes    int
num_edges    int
```

语义：

- rank-local temporal neighbor sampling index。
- 按 dst node 分组的 CSC-like temporal adjacency。
- `indptr[v]:indptr[v+1]` 对应 dst node `v` 的历史 src 列表。
- CTDG batch 的 roots 来自 canonical events；temporal index 只服务 native sampler 查历史邻居。

### `placement.pth` 与 indices

构建函数：`_build_placement_artifact()` in `starry_unigraph/preprocess/chunk.py`

`indices/master_dist_index.pth`：

- authoritative packed node index。
- 写操作、memory sync、checkpoint 使用。

`indices/read_dist_index_{rank}.pth`：

- 当前 rank 的 readable packed index。
- CTDG sampled fetch 先查本 rank `read_dist_index.cached`。

Packed DistIndex：`starry_unigraph/backends/chunk/data/dist_index.py`

```text
bits 0..47    local row id
bit  48       shared/hot flag
bit  49       cached/local-readable flag
bits 50..65   partition id
```

核心函数：

```text
encode_dist_index(local_ids, part_ids, shared=False, cached=False)
dist_index_loc(index)
dist_index_part(index)
dist_index_is_shared(index)
dist_index_is_cached(index)
```

### `partitions/part_{rank}.pth`

类型：`PartitionData` in `starry_unigraph/backends/chunk/data/partition.py`

字段：

```text
src_ids       TensorData[int64]       # remote/read src global ids per snapshot
dst_ids       TensorData[int64]       # local dst global ids per snapshot
edge_ids      TensorData[int64]
edge_src      TensorData[int32/int64] # compact src rows
edge_dst      TensorData[int32/int64] # compact dst rows/sentinel
edge_ptr      TensorData[int64]       # CSC ptr by compact dst row
dst_chunk     TensorData[int32/int64]
node_data     dict[str, TensorData]
edge_data     dict[str, TensorData]
routes        Optional[RouteData]
node_to_chunk Optional[Tensor]
```

主要方法：

- `PartitionData.from_edge_events()`：edge events -> per-rank compact graph。
- `PartitionData.to_block(snapshot_index, keep_ids=True)`：CSC-backed DGL block。
- `PartitionData.edge_events()` / `to_edge_index()`：取回 edge event/global ids。

## Runtime 装载流程

入口：`ChunkRuntimeLoader.from_prepared_artifacts()` in `starry_unigraph/backends/chunk/runtime/loader.py`

输入：

```text
prepared_dir
rank
world_size
device
config
```

输出：`ChunkRuntimeLoader`

装载流程：

```text
load PartitionData
load mem_routes_{rank}.pth
load routes/propagation_routes_{rank}.pth
load meta.json / artifact_manifest.json / partitions/manifest.json
load ChunkPlacement
load sampling/temporal_index_part_{rank}.pth
load events/canonical_events.pth
load targets/node_targets.pth
build ChunkGraphStore
build MemShareEventEngine
build NegativeSamplerHook
build task adapter
build CommPipeline
```

核心 loader 函数：

- `_load_placement()`：读取 `placement.pth`、`chunks/chunk_assignment.pth`、packed indices。
- `_load_temporal_index()`：读取 rank-local temporal sampling index。
- `_load_canonical_events()`：读取 global canonical event table。
- `_load_canonical_targets()`：读取 node target table。
- `_load_propagation_routes()`：读取 DTDG model-layer routes。
- `_load_route_list()`：读取 memory routes 或 legacy spatial routes。

## Runtime Data Views

### `ChunkGraphStore`

文件：`starry_unigraph/backends/chunk/data/graph_store.py`

字段：

```text
part                  PartitionData
placement             ChunkPlacement
_temporal_index_cache Optional[TemporalIndexView]
_event_cache          Optional[TemporalEventTable]
_target_cache         Optional[TemporalTargetTable]
```

核心方法：

- `temporal_index_view()`：从 `PartitionData` 构建或返回 cached `TemporalIndexView`。
- `temporal_events()`：返回 `TemporalEventTable`。
- `event_indices_for_snapshot(snapshot_idx, owner_part)`：按 time slice 和 event_owner 过滤本 rank events。
- `ctdg_input_view()`：构建 `EventView`，作为 CTDG `ExecutionUnit.payload`。
- `dtdg_input_view()`：构建 `SnapshotView`。

### `TemporalEventTable`

```text
src                  Tensor[E]
dst                  Tensor[E]
ts                   Tensor[E]
edge_ids             Tensor[E]
snapshot_event_ptr   int64[T+1]
event_owner          Optional[Tensor[E]]
event_dst_chunk      Optional[Tensor[E]]
event_split          Optional[Tensor[E]]
```

### `TemporalIndexView`

定义：`starry_unigraph/backends/chunk/data/plans.py`

```text
indptr       Tensor[num_nodes + 1]
indices      Tensor[num_edges]
edge_ids     Tensor[num_edges]
timestamps   Optional[Tensor[num_edges]]
placement    PlacementView
```

### `ExecutionUnit`

定义：`starry_unigraph/backends/chunk/data/plans.py`

```text
mode                "ctdg" | "dtdg"
block_id            int
placement_version   int
payload             EventView | SnapshotView | BatchData
comm_plan           CommPlanBundle
profile_hint        dict[str, float]
```

## CTDG Runtime 流程

推荐入口：

- `ChunkRuntimeLoader.iter_train_units()` / `iter_eval_units()` / `iter_predict_units()`
- `ChunkRuntimeLoader.run_train_unit_step(runtime, unit)`
- `ChunkRuntimeLoader.run_eval_unit_step(runtime, unit)`

算法流程：

```text
loader.iter_train_units(split)
  -> MemShareEventEngine.iter_units(split_slices)
     -> graph_store.event_indices_for_snapshot(t, owner_part=rank)
     -> graph_store.ctdg_input_view(...)
     -> ExecutionUnit(mode="ctdg", payload=EventView, comm_plan=base state sync)

run_train_unit_step(unit)
  -> _batch_from_sampled_unit(unit)
     -> pos_src/pos_dst from canonical events
     -> neg_src/neg_dst from NegativeSamplerHook
     -> required_nodes = unique(pos_src, pos_dst, neg_src, neg_dst)
     -> event_engine.sample(unit, extra_root_nodes=neg_dst, required_nodes=required_nodes)
     -> split sampled input nodes through read_dist_index cached bit
     -> _make_dynamic_fetch_plan(remote_read_index, local_read_index)
     -> BatchData(mfgs, node_ids, id_map_nodes, pos/neg edges, remote_manifest)
  -> run_train_step(batch)
     -> _execute_and_patch_fetch(runtime, batch)
        -> CommPipeline.submit_fetch(FetchPlan)
        -> patch fetched feature/memory into MFG srcdata
     -> run_batch(model, batch, task_adapter, optimizer, train=True)
     -> optional submit_memory_update()
```

### `MemShareEventEngine`

文件：`starry_unigraph/backends/chunk/runtime/event_engine.py`

核心函数：

- `from_config(graph_store, cfg)`：构建 engine。
- `iter_units(split_slices, plans_fn)`：生成 CTDG `ExecutionUnit`。
- `sample(unit, extra_root_nodes=None, extra_root_ts=None, required_nodes=None)`：调用 native sampler 并生成 `CTDGSampleResult`。
- `_read_index_for_nodes(nodes, placement, local_part)`：用 `read_dist_index` 或 `master_dist_index` 拆分 local/remote read set。

`CTDGSampleResult`：

```text
mfgs
input_nodes
output_nodes
edge_ids
node_ts
edge_ts
memory_node_ids
remote_node_ids
local_node_ids
remote_read_index
local_read_index
id_map_nodes
```

### CTDG FetchPlan

构建函数：`ChunkRuntimeLoader._make_dynamic_fetch_plan()` in `runtime/loader.py`

输入：

```text
remote_read_index       packed DistIndex[R]
local_read_index        packed DistIndex[L]
remote_node_ids         Optional[int64[R]]
local_node_ids          Optional[int64[L]]
```

输出：`FetchPlan` in `data/plans.py`

```text
block_id
placement_version
feature_node_ids        empty in packed fetch path
feature_owners          dist_index_part(remote_read_index)
remote_read_index
local_read_index
remote_node_ids
local_node_ids
memory_read_index       remote_read_index
cache_policy            "sampled_packed_dist_index"
```

## DTDG Runtime 流程

入口：

- `ChunkRuntimeLoader.iter_train()` / `iter_eval()` / `iter_predict()` 返回 `BatchData`。
- `ChunkRuntimeLoader.iter_train_envelopes()` 返回带 `PropagationPlan` 的 `ExecutionUnit`。

算法流程：

```text
PartitionData snapshot
  -> task_adapter.build_batch()
     -> PartitionData.to_block(snapshot_idx, keep_ids=True)
     -> BatchData(mfgs=DGLBlock, node_ids, labels/edges)
  -> _make_comm_plan(block_id)
     -> optional legacy spatial fetch plan
     -> StateSyncPlan from MemoryRouteData
     -> PropagationPlan from routes/propagation_routes_{rank}.pth
  -> model layer uses ChunkPropagationRoute.forward/async_forward()
  -> loss.backward()
  -> route backward all-to-all returns remote grad and index_add_ into local x
```

## Task 层

### `BatchData`

文件：`starry_unigraph/backends/chunk/data/batch.py`

```text
mfgs              DGLBlock | list | backend-native blocks
node_ids          int64[N]
id_map_nodes      Optional[int64[N]]  # compact embedding rows 对应的 global ids
pos_src/pos_dst   Optional[int64[M]]
neg_src/neg_dst   Optional[int64[M]]
labels            Optional[Tensor]
target_nodes      Optional[int64[K]]
timestamps        Optional[Tensor]
chunk_id          Optional[Any]
local_node_mask   Optional[bool[N]]
remote_manifest   Optional[dict]
```

### Task adapters

文件：`starry_unigraph/backends/chunk/runtime/task_adapter.py`

核心接口：

```text
ChunkTaskAdapter.build_batch(part, snapshot_idx, event_pos, split, neg_sampler, num_nodes) -> BatchData
ChunkTaskAdapter.compute_loss(model_output, batch) -> Tensor
ChunkTaskAdapter.compute_metrics(model_output, batch) -> dict
```

实现：

- `EdgePredictAdapter`
- `EdgeRegressAdapter`
- `NodeClassifyAdapter`
- `NodeRegressAdapter`
- `get_task_adapter(task_type, **kwargs)`

CTDG native sampled path不主要依赖 `EdgePredictAdapter.build_batch()` 构造 MFG，而是由 `_batch_from_sampled_unit()` 直接将 native sampler 输出包装成 `BatchData`；loss/metric 仍复用 task adapter。

### Negative sampler

文件：`starry_unigraph/backends/chunk/runtime/sampler.py`

类：`EdgePredictNegativeSampler`

配置入口：`NegativeSamplerHook.from_config(cfg)`

输入：

```text
pos_src, pos_dst     Tensor[M]
num_nodes            int
neg_ratio            int
split                "train" | "val" | "test"
```

输出：

```text
neg_src, neg_dst     Tensor[M * neg_ratio]
```

策略：

- train：从 `local_dst_pool` 采样，按 `train_remote_dst_prob` 混入 `global_dst_pool`。
- val/test：默认 `test_policy="global_average"`，从 global pool 采样。

### Model heads and id map

文件：

- `starry_unigraph/models/task_head.py`
- `starry_unigraph/models/id_map.py`

约定：

- batch 内 edge/node ids 可以是 global ids。
- 如果 `batch.id_map_nodes` 存在，head 必须通过 `compact_rows(node_ids, id_map_nodes, embeddings)` 映射到 compact row。
- 只有 `id_map_nodes is None` 时才允许把输入 ids 当作 embedding row ids。

## 通信层

### Comm plans

文件：`starry_unigraph/backends/chunk/data/plans.py`

```text
FetchPlan
  remote_read_index / local_read_index / feature_owners / memory_read_index

PropagationPlan
  layer_routes: list[ChunkPropagationRoute]
  autograd_enabled

StateSyncPlan
  update_node_ids
  update_owners
  update_index
  replica_node_ids
  replica_index

CommPlanBundle
  fetch
  propagation
  state_sync
```

### CommPipeline

文件：`starry_unigraph/backends/chunk/data/comm.py`

核心接口：

```text
submit_fetch(plan, feature_rows=None, memory_rows=None) -> CommHandle("fetch")
await_fetch() -> FetchResult

submit_spatial(route, features) -> CommHandle("spatial")       # legacy SpatialRouteData path
await_spatial() -> SpatialResult

submit_memory(route, memory, ts, baseline_memory=None, ...) -> CommHandle("memory")
await_memory() -> MemoryResult

submit_replica(route, memory) -> Optional[CommHandle("replica")]
await_replica() -> Optional[(recv_node_ids, recv_memory)]

await_handle(handle)
drain_all_sync()
```

CTDG sampled-fetch path使用 `submit_fetch()`：

- local rows：`dist_index_loc(local_read_index)` 本地 gather。
- remote rows：先 all-to-all 请求 loc，再 all-to-all 返回 feature/memory values。

Memory sync path使用 `submit_memory()`：

- 可用 `baseline_memory` 和 `change_threshold` 过滤小变化更新。
- `unique_index/recv_index` 为 packed master DistIndex，用于新热路径；`unique_nodes/recv_node_ids` 保留兼容。

## DTDG Propagation Route

文件：`starry_unigraph/models/layers/route.py`

数据结构：

```text
ChunkPropagationRoute
  send_sizes      list[int]
  recv_sizes      list[int]
  send_index      Optional[int64[S]]
  recv_src_rows   Optional[int64[R]]
  append_recv     bool
```

forward：

```text
send_buf = x[send_index]
dist.all_to_all_single(async_op=True)
recv()
if recv_src_rows: reorder recv_buf
if append_recv: return cat([local_x, recv_buf])
else: return recv_buf
```

backward：

```text
split grad_output into local_grad and remote_grad
reverse all_to_all_single(remote_grad)
grad_x = zeros_like(local_x)
grad_x.index_add_(0, send_index, received_grad)
grad_x += local_grad
```

模型层必须在真正需要 remote activation 的位置调用 `route.forward()` 或 `await route.async_forward()`，不要在 dataloader 或 train step 外层提前执行。

## Gradient Version Manager

文件：`starry_unigraph/backends/chunk/runtime/weight_version_manager.py`

核心类型：

```text
GradBucket(version, grads, work, ready)
WeightVersionManager
```

核心接口：

```text
save_param_backup(version)
restore_param_backup(version)
submit_grad_bucket(version, grads) -> GradBucket
maybe_apply_ready(optimizer) -> bool
drain_all(optimizer) -> int
describe() -> dict
```

语义：

- step `v` forward 前保存参数 flat backup 到 `v % num_backup_slots`。
- backward 后将 grads flatten，并用 async all_reduce 提交。
- optimizer step 按 version 顺序 apply，即使 all_reduce 完成顺序不同。

## 端到端数据流

### CTDG sampled edge predict

```text
Prepare:
  raw_events
    -> canonical_events.pth
    -> chunk assignment + placement + DistIndex
    -> part_{rank}.pth
    -> temporal_index_part_{rank}.pth
    -> mem_routes_{rank}.pth

Runtime:
  canonical_events.time_ptr[t:t+1]
    -> event_indices filtered by event_owner == rank
    -> pos_src/pos_dst
    -> EdgePredictNegativeSampler -> neg_src/neg_dst
    -> roots = pos_src + pos_dst + neg_dst
    -> MemShareEventEngine.sample()
    -> CTDGSampleResult.remote/local_read_index
    -> FetchPlan(sampled_packed_dist_index)
    -> CommPipeline.submit_fetch()
    -> patch fetched feature/memory into mfgs
    -> model(batch)
    -> task_adapter.compute_loss/metrics
    -> optional submit_memory_update()
```

### DTDG snapshot execution

```text
Prepare:
  raw/snapshot events
    -> PartitionData per rank
    -> propagation_routes_{rank}.pth
    -> mem_routes_{rank}.pth

Runtime:
  PartitionData.to_block(snapshot)
    -> BatchData(mfgs=DGLBlock)
    -> PropagationPlan(layer_routes)
    -> model layer calls ChunkPropagationRoute
    -> task head maps ids through id_map if present
    -> loss/metrics
```

## 当前兼容与限制

- `placement.pth` 仍写入并读取，用于兼容旧 loader 和调试。
- `node_owner/node_master` 是 placement/debug 字段；新增热路径应优先使用 packed DistIndex。
- CTDG prebuilt `SpatialRouteData` 是 legacy path，不是主链路。
- `partitions/part_{rank}.pth` 仍是 snapshot path 和 temporal index 构建的 per-rank graph container。
- `routes/manifest.json` 记录 DTDG propagation route policy；构建失败时可配置 empty fallback。

## 2026-05-18 修复记录

基于 code review，完成以下 P1 级别修复：

### 1. Memory Timestamp Reduce ✅
- **文件**: `backends/chunk/data/comm.py:464-502`
- **改动**: 在 `await_memory()` 中添加 timestamp reduce
- **逻辑**: 同一节点多个更新时，保留最新 timestamp 的更新
- **影响**: 修复乱序更新可能覆盖新 memory 的问题

### 2. 负采样 dst pool ✅
- **文件**: `backends/chunk/runtime/loader.py:266-283`
- **改动**: 从 `canonical_events.dst` 提取 dst pool（而非 owner pool）
- **Fallback**: 如果 canonical_events 不存在，使用 owner pool（兼容性）
- **影响**: 修复 train/test 负样本分布不一致问题

### 3. CTDG 自适应切分配置 ✅
- **文件**: `preprocess/chunk.py:616-632`
- **改动**: 添加 `time_slice_strategy` 配置项
- **支持**: `"adaptive"` (C++ native) 或 `"snapshot"` (legacy)
- **配置示例**:
  ```yaml
  chunk:
    time_slice_strategy: "adaptive"
    target_batch_size: 200
    graph_feature: 1.0
    alpha: 1.0
    beta: 0.5
  ```

### 4. DTDG ChunkPropagationRoute Snapshot 语义 ✅
- **文件**: `backends/chunk/prepare/propagation_builder.py:18-180`
- **新增**: `build_propagation_routes_from_snapshots()` 函数
- **支持**: 
  - 使用 `dst_ids/src_ids` 从 snapshot 构建
  - 生成 `recv_src_rows` 用于 compact row indexing
  - 对齐 DTDG block 的 `[dst, remote_src]` row layout
- **使用**: 新代码应使用此函数，旧函数 `build_propagation_routes()` 保留作为 fallback

### 待完善项

- [ ] Chunk Graph Metis 集成（当前为贪心算法）
- [ ] Runtime 性能约束审查（确保无 edge/event 级 Python loop）
- [ ] CTDG fetch route 缓存优化

详细修复信息见 `FIXES_NEEDED.md`。

---

## 2026-05-19 模型层实现

完成 CTDG 和 DTDG 完整训练链路，补齐缺失的模型层和消息传递机制。

### CTDG 模块（参考 MemShare-public）

#### 1. Mailbox (`backends/chunk/runtime/mailbox.py`)

历史消息缓存，支持分布式训练。

**核心 API**:
```python
from starry_unigraph.backends.chunk.runtime.mailbox import Mailbox, MailboxConfig

# 初始化
config = MailboxConfig(
    num_nodes=10000,
    memory_dim=100,
    cache_size=10,
    device="cuda",
    deliver_to="self",  # "self" 或 "neighbors"
)
mailbox = Mailbox(config)

# 存储消息
mailbox.push(node_ids, messages, timestamps)

# 获取消息（支持时间戳过滤和聚合）
messages, ts = mailbox.get(
    node_ids,
    before_timestamp=current_ts,
    aggregation="last",  # "last", "mean", "max"
)

# 更新 memory cache
mailbox.set_memory_local(node_ids, memory, timestamps, reduce_op="max")
```

#### 2. MemoryUpdater (`backends/chunk/model/memory_updater.py`)

通过消息更新节点 memory。

**支持的更新器**:
```python
from starry_unigraph.backends.chunk.model.memory_updater import (
    GRUMemoryUpdater,
    RNNMemoryUpdater,
    TransformerMemoryUpdater,
)

# GRU 更新器
updater = GRUMemoryUpdater(
    memory_dim=100,
    message_dim=300,  # 2 * memory_dim + edge_dim
    time_dim=100,
    node_feat_dim=172,
    combine_node_feature=True,
)

# 从 DGL block 更新
new_memory = updater.forward_from_mfg(mfg)

# 或直接调用
new_memory = updater.forward(
    memory=old_memory,
    messages=aggregated_messages,
    timestamps=current_ts,
    memory_ts=last_update_ts,
    node_feats=node_features,
)
```

#### 3. CTDG Layers (`backends/chunk/model/ctdg_layers.py`)

Temporal GNN 层。

**TransformerAttentionLayer**:
```python
from starry_unigraph.backends.chunk.model.ctdg_layers import TransformerAttentionLayer

layer = TransformerAttentionLayer(
    dim_node_feat=100,
    dim_edge_feat=172,
    dim_time=100,
    num_head=2,
    dropout=0.1,
    att_dropout=0.1,
    dim_out=100,
    combined=False,
)

# 输入 DGL block，期望 srcdata 包含:
# - 'h': node features [num_src, dim_node_feat]
# - 'f': edge features [num_edges, dim_edge_feat] (edata)
# - 'dt': time delta [num_edges] (edata)
output = layer(block)  # [num_dst, dim_out]
```

**其他层**:
```python
from starry_unigraph.backends.chunk.model.ctdg_layers import (
    IdentityNormLayer,      # JODIE 用
    JODIETimeEmbedding,     # 时间调制
    EdgePredictor,          # 边预测
)
```

#### 4. GeneralModel (`backends/chunk/model/general_model.py`)

完整的 CTDG 模型（TGN/JODIE/DyRep）。

**使用示例**:
```python
from starry_unigraph.backends.chunk.model import GeneralModel

model = GeneralModel(
    dim_node=172,
    dim_edge=172,
    sample_param={'history': 1},
    memory_param={
        'type': 'node',
        'dim_out': 100,
        'memory_update': 'gru',  # 'gru', 'rnn', 'transformer'
        'dim_time': 100,
        'combine_node_feature': True,
    },
    gnn_param={
        'arch': 'transformer_attention',  # 或 'identity'
        'layer': 2,
        'dim_time': 100,
        'att_head': 2,
        'dim_out': 100,
        'time_transform': None,  # 或 'JODIE'
        'dyrep': False,
    },
    train_param={'dropout': 0.1, 'att_dropout': 0.1},
    num_nodes=10000,
    mailbox=mailbox,
)

# 训练
pos_scores, neg_scores = model(
    mfgs,  # mfgs[layer][history] = DGL block
    metadata={
        'src_pos_index': pos_src_idx,
        'dst_pos_index': pos_dst_idx,
        'dst_neg_index': neg_dst_idx,
    },
    neg_samples=1,
    mode="triplet",
)
```

**配置工厂**:
```python
model = GeneralModel.from_config(
    dim_node=172,
    dim_edge=172,
    num_nodes=10000,
    config={
        'sample': {'history': 1},
        'memory': {...},
        'gnn': {...},
        'train': {...},
    },
    mailbox=mailbox,
)
```

### DTDG 模块（参考 FlareDTDG）

#### 5. Async GNN Encoders (`models/encoders/async_gnn.py`)

支持异步通信/计算 overlap 的 GNN 层。

**AsyncModule 基类**:
```python
from starry_unigraph.models.encoders import AsyncModule

class MyGNN(AsyncModule):
    async def async_forward(self, g, x):
        # 异步 forward
        await self.yield_forward()  # yield control
        return output
    
    def forward(self, g, x):
        # 同步 wrapper
        return asyncio.run(self.async_forward(g, x))
```

**GCN / GraphSAGE / GAT**:
```python
from starry_unigraph.models.encoders import GCN, GraphSAGE, GAT

# GCN
gcn = GCN(
    in_features=172,
    out_features=100,
    num_layers=2,
    bias=True,
    shortcut=False,
)

# 同步调用
output = gcn(block, x)

# 异步调用（支持 ChunkPropagationRoute）
output = await gcn.async_forward(block, x)
```

#### 6. STGraphLoader (`backends/chunk/runtime/stg_loader.py`)

时空图加载器，支持 chunk decay 策略和 RNN state 管理。

**核心概念**:
- **Chunk Decay**: 历史层按 chunk 优先级截断，最近的 chunk 保留更多节点
- **RNN State**: 跨 snapshot 管理 RNN hidden state（pad/mix 模式）
- **STGraphBlob**: 历史 snapshot 窗口，每个 graph 带 state 管理方法

**使用示例**:
```python
from starry_unigraph.backends.chunk.runtime.stg_loader import STGraphLoader

loader = STGraphLoader(
    partition_data=part_data,
    chunk_assignment=chunk_assignment,
    device="cuda",
    num_full_snaps=1,
    chunk_decay=[9, 8, 7, 6, 5, 4, 3, 2, 1, 0],  # 最近的优先
    rnn_state_mode="mix",  # "pad" 或 "mix"
)

# 迭代训练
for blob in loader():
    # blob 是 STGraphBlob，包含历史 snapshot 窗口
    for g in blob:
        # g 是 DGLBlock，带 state 管理方法
        state = g.flare_fetch_state(state)
        output = model(g, state)
        g.flare_store_state(output)
```

**Chunk Decay 策略**:
```python
# 假设 10 个 chunks，chunk_decay = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
# chunk_ptr = [0, 100, 250, 450, 700, 1000, 1350, 1750, 2200, 2700, 3000]

# Layer 0 (最旧): 只保留 chunk 9
#   endpoint = chunk_ptr[9+1] = 3000 节点

# Layer 1: 保留 chunk 9 + 8
#   endpoint = chunk_ptr[8+1] = 2700 节点

# Layer 2: 保留 chunk 9 + 8 + 7
#   endpoint = chunk_ptr[7+1] = 2200 节点

# ...

# Full snapshot: 保留所有节点
#   endpoint = None
```

**Runtime chunk_order override**:
```python
# 动态调整 chunk 优先级（负载均衡）
for blob in loader(chunk_order=dynamic_order):
    # chunk_order[c] = chunk c 的优先级 rank
    # loader 会根据 chunk_order 重新计算 endpoints
    ...
```

**RNN State 管理**:
```python
# Pad mode: 每次都 zero-pad state
loader = STGraphLoader(..., rnn_state_mode="pad")

# Mix mode: 混合当前 state 和历史 state
loader = STGraphLoader(..., rnn_state_mode="mix")

# 在模型中使用
for blob in loader():
    for g in blob:
        # Fetch state (自动 pad/mix)
        state = g.flare_fetch_state(prev_state)
        
        # Forward
        output, new_state = rnn_model(g, state)
        
        # Store state
        g.flare_store_state(new_state)
```

### 端到端训练示例

#### CTDG (TGN)
```python
from starry_unigraph.backends.chunk.model import GeneralModel
from starry_unigraph.backends.chunk.runtime.mailbox import Mailbox, MailboxConfig

# 1. 初始化 mailbox
mailbox = Mailbox(MailboxConfig(
    num_nodes=num_nodes,
    memory_dim=100,
    cache_size=10,
    device="cuda",
))

# 2. 初始化模型
model = GeneralModel(
    dim_node=172,
    dim_edge=172,
    sample_param={'history': 1},
    memory_param={
        'type': 'node',
        'dim_out': 100,
        'memory_update': 'gru',
        'dim_time': 100,
        'combine_node_feature': True,
    },
    gnn_param={
        'arch': 'transformer_attention',
        'layer': 2,
        'dim_time': 100,
        'att_head': 2,
        'dim_out': 100,
    },
    train_param={'dropout': 0.1, 'att_dropout': 0.1},
    num_nodes=num_nodes,
    mailbox=mailbox,
)

# 3. 训练循环
for epoch in range(num_epochs):
    for batch in dataloader:
        # batch 包含 mfgs 和 metadata
        pos_scores, neg_scores = model(
            batch.mfgs,
            batch.metadata,
            neg_samples=1,
        )
        
        loss = criterion(pos_scores, neg_scores)
        loss.backward()
        optimizer.step()
```

#### DTDG (GCN + STGraphLoader)
```python
from starry_unigraph.models.encoders import GCN
from starry_unigraph.backends.chunk.runtime.stg_loader import STGraphLoader

# 1. 初始化 loader
loader = STGraphLoader(
    partition_data=part_data,
    chunk_assignment=chunk_assignment,
    device="cuda",
    num_full_snaps=1,
    chunk_decay=[9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
    rnn_state_mode="mix",
)

# 2. 初始化模型
gcn = GCN(in_features=172, out_features=100, num_layers=2)

# 3. 训练循环
for blob in loader():
    for g in blob:
        # Fetch RNN state
        state = g.flare_fetch_state(state)
        
        # GCN forward
        output = gcn(g, state)
        
        # Task head
        pred = task_head(output)
        loss = criterion(pred, labels)
        loss.backward()
        optimizer.step()
        
        # Store RNN state
        g.flare_store_state(output)
```

### 关键设计决策

#### Chunk Decay: CSR vs ind2ptr

**FlareDTDG 原始方法**:
```python
# 需要 torch_sparse.ind2ptr
inds, perm = chunk_order.sort()
ends = torch_sparse.ind2ptr(inds, num_chunks)
```

**当前实现（改进）**:
```python
# 直接使用 chunk_ptr CSR
chunk_ptr = chunk_assignment.chunk_ptr
end = chunk_ptr[chunk_id + 1]
```

**优势**:
- ✅ 无需 `torch_sparse` 依赖
- ✅ O(1) 查询，无需排序
- ✅ 内存高效
- ✅ 支持 runtime 动态覆盖

详见 `CHUNK_DECAY_COMPARISON.md`。

### 模块状态

以下状态以 2026-05-19 code review 为准。旧的“已支持”判断必须按“是否接入主训练链路”而不是“文件是否存在”判定。

| 模块 | 当前状态 | 文件 | 结论 |
|------|----------|------|------|
| ChunkRuntimeLoader BatchData path | 部分支持 | `backends/chunk/runtime/loader.py` | `PartitionData.to_block()` 可生成 `BatchData.mfgs`，但默认模型是 `SimpleChunkModel`，不消费图结构。 |
| STGraphLoader / STGraphBlob | 未接入主链路 | `backends/chunk/runtime/stg_loader.py` | 文件可作为实验入口，但 `ChunkRuntimeLoader` 仍不 yield `STGraphBlob`，`SchedulerSession` 的 chunk 分支也不走 Flare-style `model(blob)`。 |
| FlareDTDG train entry compatibility | 不支持 | `examples/train_mpnn_lstm_4gpu.py`, `session.py` | 现有 train 入口走 `SchedulerSession.run_epoch()` 和 `BatchData`，不兼容 `~/FlareDTDG/run_flare2.py` 的 `STGraphLoader(...)(chunk_order, chunk_decay, num_full_snaps, disable_states, disable_routes)` 训练协议。 |
| ChunkPropagationRoute | 部分支持 | `models/layers/route.py` | route 类型和 prepare artifact 已有，但需要模型层从 DGL graph/block 的 `route` 字段显式调用；默认 chunk 模型未调用。 |
| Async GNN | 实验性 | `models/encoders/async_gnn.py` | 提供 FlareDTDG-style route-aware GNN 层，但未被 chunk runtime 默认模型构建/训练入口使用。 |
| CTDG sampled edge predict | 部分支持 | `backends/chunk/runtime/event_engine.py`, `loader.py` | canonical events + native sampler + dynamic fetch path 存在；仅 edge/link predict 进入该热路径。 |
| CTDG node predict | 不支持 sampled 主链路 | `backends/chunk/runtime/loader.py` | target store 和 task adapter 可用，但没有 `target_node -> temporal sampler -> dynamic fetch -> node loss` 的 sampled CTDG node predict 链路。 |
| MemShare SharedMailBox equivalent | 不支持 | `backends/chunk/runtime/mailbox.py` | 当前是 `dict[int, deque]` placeholder，不是 tensorized K-slot mailbox，不支持 shared node/historical cache/p2p mail/memory sync。 |
| MemoryUpdater | 部分支持 | `backends/chunk/model/memory_updater.py` | GRU/RNN/Transformer updater 类存在，但依赖 `mfg.srcdata['mem_input','mail_ts','mem_ts']`；chunk sampled fetch 当前没有完整 patch mailbox 输入。 |
| GeneralModel | 未接入主链路 | `backends/chunk/model/general_model.py` | 接近 MemShare `GeneralModel` 形态，但 `ChunkRuntimeLoader.build_default_model()` 不构建它，`run_batch()` 调用的是 `model(batch)`，而 MemShare 需要 `model(mfgs, metadata, async_param=...)`。 |

## 2026-05-19 Code Review：FlareDTDG 与 MemShare 兼容性

### 1. FlareDTDG `train_*.py` / `run_flare2.py` 入口对照

参照文件：

- `~/FlareDTDG/run_flare2.py`
- `~/FlareDTDG/flare2/data/stc_loader.py`
- `~/FlareDTDG/flare2/nn/light/tgcn.py`
- `~/FlareDTDG/flare2/nn/light/mpnn_lstm.py`
- `~/FlareDTDG/flare2/nn/light/evolvegcn.py`

FlareDTDG 训练入口需要以下协议：

```text
PartitionData.load(...)
  -> chunk_index = data.pop_ndata("c")[0].item()
  -> STGraphLoader.from_partition_data(data, device, chunk_index)
  -> train_loader = loader[:train_end]
  -> for blob in train_loader(
         chunk_order=...,
         chunk_decay=...,
         num_full_snaps=...,
         disable_states=...,
         disable_routes=...):
       preds, state = model(blob)
       loss(preds, blob)
```

模型层依赖：

```text
STGraphBlob.__iter__()
  -> graph.flare_fetch_state(...)
  -> graph.flare_apply_route(...) / graph.flare_async_route(...)
  -> graph.flare_store_state(...)
```

当前 chunk 链路差异：

- `ChunkRuntimeLoader.iter_train()` 仍返回 `BatchData`，不是 `STGraphBlob`。
- `SchedulerSession` 的 `graph_mode == "chunk"` 分支调用 `ChunkRuntimeLoader.build_default_model()`，默认模型是 `SimpleChunkModel`。
- `SimpleChunkModel.forward(batch)` 只做 embedding lookup/head，不消费 DGL block，也不调用 `flare_fetch_state`、`flare_store_state` 或 `ChunkPropagationRoute`。
- `backends/chunk/runtime/stg_loader.py` 的 API 不是 FlareDTDG 入口等价替换：当前 `from_partition_data()` 接收 `chunk_assignment`，不是 Flare 的 `chunk_index`；`__call__()` 当前只接收 `chunk_order` 和 `disable_states`，不接收 Flare 入口使用的 `chunk_decay`、`num_full_snaps`、`disable_routes` runtime 参数。
- chunk `PartitionData` artifact 需要确认保存多 snapshot/window。若 prepare 仍只写单 snapshot，STGraphLoader 即使接入也无法提供 FlareDTDG 的历史快照序列。

必须补齐的组件：

1. `ChunkRuntimeLoader.iter_train_stgraph()` 或配置开关，使 chunk DTDG path 可直接 yield `STGraphBlob`。
2. `SchedulerSession` chunk 分支识别 DTDG + STGLoader 模式，并调用 `run_flare_train_step` 等价逻辑或 chunk-native `run_stgraph_train_step`。
3. `STGraphLoader.from_partition_data(data, device, chunk_index=...)` 兼容 FlareDTDG 参数，同时保留 chunk_assignment 快路径。
4. `STGraphLoader.__call__(chunk_order, chunk_decay, num_full_snaps, disable_states, disable_routes)` 与 FlareDTDG 入口对齐。
5. 正式模型构建不能默认 `SimpleChunkModel`，需要接入 `models/encoders/async_gnn.py` 或 DTDG Flare 模型，使模型消费 `STGraphBlob/DGLBlock`。

### 2. MemShare `examples/train_boundery.py` 入口对照

参照文件：

- `~/MemShare-public/MemShare/examples/train_boundery.py`
- `~/MemShare-public/MemShare/starrygl/sample/data_loader.py`
- `~/MemShare-public/MemShare/starrygl/sample/memory/shared_mailbox.py`
- `~/MemShare-public/MemShare/starrygl/module/modules.py`
- `~/MemShare-public/MemShare/starrygl/module/memorys.py`

MemShare 训练入口需要以下协议：

```text
load_from_speed(...)
  -> DistributedGraphStore / TemporalNeighborSampleGraph
  -> SharedMailBox(num_nodes, memory_param, dim_edge_feat, shared_nodes_index, ada_param)
  -> NeighborSampler(policy=boundery_recent_uniform|boundery_recent_decay|recent, ...)
  -> LocalNegativeSampling(...)
  -> DistributedDataLoader(..., mailbox=mailbox, is_pipeline=True)
  -> for roots, mfgs, metadata in trainloader:
       edge_feats = graph.get_local_efeat(...)
       async_param = (update_mail, src, dst, ts, edge_feats, trainloader.async_feature)
       pred_pos, pred_neg = GeneralModel(mfgs, metadata, neg_samples, async_param=async_param)
       loss.backward()
       optimizer.step()
       mailbox.update_shared()
       mailbox.update_p2p_mem()
       mailbox.update_p2p_mail()
```

必要数据字段：

```text
mfg.srcdata["ID"]
mfg.srcdata["mem"]
mfg.srcdata["mem_ts"]
mfg.srcdata["mem_input"]
mfg.srcdata["mail_ts"]
mfg.srcdata["ts"]
mfg.srcdata["h"]
metadata["src_pos_index"]
metadata["dst_pos_index"]
metadata["dst_neg_index"]
metadata["train_ratio_pos"] / ["train_ratio_neg"] for mixed local/global negative loss
```

当前 chunk 链路差异：

- `MemShareEventEngine` 只覆盖 native temporal sampler 输出和 sampled read set，不提供 MemShare `DistributedDataLoader` 的 `(roots, mfgs, metadata)` 完整协议。
- `_batch_from_sampled_unit()` 生成 `BatchData(pos_src,pos_dst,neg_src,neg_dst,remote_manifest)`，不是 MemShare `metadata`，也没有 `roots.eids/roots.ts` 对象。
- `CommPipeline.submit_fetch()` 可 patch feature/memory，但没有 mailbox fetch：没有 patch `mem_input`、`mail_ts`、`his_mem`、`his_ts`。
- `backends/chunk/runtime/mailbox.py` 不是 `SharedMailBox` 等价实现：没有 `DistributedTensor`、`shared_nodes_index`、`next_mail_pos` tensor K-slot 写入、`historical_cache`、`set_memory_all_reduce()`、`update_shared()`、`update_p2p_mem()`、`update_p2p_mail()`。
- `backends/chunk/model/general_model.py` 的 forward signature 是 `forward(mfgs, metadata, ...)`，但 chunk `run_batch()` 固定调用 `model(batch)`；所以即使模型存在，也没有被当前 runtime 正确调用。
- `AsyncMemeoryUpdater` 的核心模式缺失：`all_update`、`historical`、`p2p/all_reduce/local` 的 mail/memory submit 和 delayed fetch overlap 没有接入。
- boundary sampling 策略未等价支持：`boundery_recent_uniform`、`boundery_recent_decay`、`AdaParameter` 驱动的概率/历史衰减决策没有在 chunk native sampler 中形成完整 policy。
- 负采样只覆盖 dst pool 和 local/global 混合概率的基础版本；没有 MemShare `LocalNegativeSampling(..., local_mask, ada_param)` 和 `train_ratio_pos/train_ratio_neg` loss 权重协议。

必须补齐的组件：

1. `ChunkSharedMailbox`：以 `backends/ctdg/runtime/memory.py` 或 MemShare `SharedMailBox` 为基准实现 tensorized memory + K-slot mailbox。
2. `FetchPlan`/`FetchResult` 扩展 mailbox channel：remote/local fetch 必须返回 memory、memory_ts、mailbox slots、mailbox_ts。
3. sampled MFG patch：在 model forward 前填充 `mem`、`mem_ts`、`mem_input`、`mail_ts`、`ts`、`ID`，historical 模式还要填 `his_mem`、`his_ts`。
4. `GeneralModel` runtime adapter：支持 `model(mfgs, metadata, neg_samples, async_param)`，或提供 wrapper 让 `run_batch()` 能识别 MemShare-style model。
5. `metadata` builder：从 `BatchData`/native sampler 输出构建 `src_pos_index`、`dst_pos_index`、`dst_neg_index`，并补齐 mixed negative loss 权重。
6. mailbox update lifecycle：forward 后按 `last_updated_nid/memory/ts` 构造 memory update，按 positive edges 构造 `[src_mem || dst_mem || edge_feat]` mail，执行 local write + owner/shared/p2p sync。
7. boundary sampler policy：将 `boundery_recent_uniform`、`boundery_recent_decay`、`AdaParameter`、local/remote node-part edge-part 信息纳入 native sampler 或 Python sampler wrapper。

### 当前判定

- FlareDTDG DTDG train entry：当前 chunk library **不支持**，只能说有部分低层组件。
- MemShare `train_boundery.py` CTDG train entry：当前 chunk library **不支持**，只能说有 native sampler、基础 negative sampler、部分模型类雏形。
- 下一步实现优先级应先接入两个 runtime adapter：`ChunkSTGraphRuntimeAdapter` 和 `ChunkMemShareRuntimeAdapter`，再替换默认 `SimpleChunkModel`。

## 2026-05-19 真实数据验证结果

验证数据：

```text
/mnt/data/zlj/starrygl-data/ctdg/WIKI.pth
```

验证命令：

```text
python examples/bench_chunk_native.py \
  --dataset WIKI \
  --mode prepare \
  --device cpu \
  --epochs 1 \
  --batch-size 3000 \
  --num-windows 2 \
  --fanout 5 \
  --chunks-per-partition 2 \
  --artifact-root /tmp/starry_chunk_review_wiki_prepare
```

结果：失败，prepare 阶段触发 native `Floating point exception`。

trace：

```text
dgl.partition.metis_partition_assignment
  -> backends/chunk/prepare/pipeline.py::_try_metis_partition
  -> build_node_partition
  -> prepare
  -> preprocess/chunk.py::build_partitions
```

结论：

- 当前真实数据 `chunk` prepare 不可用；DGL METIS 的 native signal 不是 Python exception，当前 `_try_metis_partition()` 的 `try/except` 无法兜底。
- 即使配置 `partition_strategy=mem_share` 且 `num_chunks_per_partition=1`，后续仍会进入 `_build_partitioned_metis_chunk_assignment()`，继续调用 DGL METIS。
- 因为 prepare 已崩溃，真实数据下无法进入 runtime/train 验证。

同时运行 chunk 单元测试：

```text
python -m pytest starry_unigraph/backends/chunk/runtime/test_runtime.py -q
```

结果：

```text
2 failed, 17 passed
```

失败点：

```text
backends/chunk/prepare/pipeline.py::_assignment_from_node_to_chunk
  TypeError: ChunkAssignment.__init__() got unexpected keyword argument 'chunk_to_nodes'

backends/chunk/prepare/pipeline.py::prepare(assignment=...)
  AttributeError: 'ChunkAssignment' object has no attribute 'chunk_to_nodes'
```

原因：

- `backends/chunk/prepare/chunk_assignment.py::ChunkAssignment` 已切换为 CSR 字段：

```text
chunk_ptr
chunk_nodes
```

- `backends/chunk/prepare/pipeline.py` 仍按旧 list 字段：

```text
chunk_to_nodes
```

构造和复制 `ChunkAssignment`。

通过的低层测试：

```text
python -m pytest \
  starry_unigraph/backends/chunk/data/test_route.py \
  starry_unigraph/backends/chunk/data/test_edge_index_conversion.py \
  starry_unigraph/backends/chunk/data/test_propagation_route.py -q

17 passed
```

当前真实支持矩阵：

| 能力 | 真实数据端到端状态 | 阻断点 |
|------|--------------------|--------|
| chunk prepare on WIKI | 不支持 | DGL METIS native SIGFPE；`ChunkAssignment` API 断裂 |
| chunk CTDG sampled edge train | 未验证成功 | prepare 阶段阻断 |
| chunk DTDG/STGraph train | 不支持 | 主入口不 yield `STGraphBlob`，默认模型不消费图结构 |
| MemShare-style mailbox train | 不支持 | 无 `SharedMailBox` 等价实现，无 mailbox fetch/patch/sync |
| low-level route/partition conversion tests | 部分支持 | data 层 17 个测试通过 |

修复优先级：

1. 修复 `pipeline.py` 的 `ChunkAssignment` CSR API 使用，删除 `chunk_to_nodes` 旧字段路径。
2. 在 prepare 中提供完全不调用 DGL METIS 的 fallback 策略，且默认真实数据 smoke 可强制走 fallback；不能依赖 Python `try/except` 捕获 native signal。
3. 真实 WIKI 单进程 prepare 通过后，再验证 `canonical_events.pth`、`temporal_index_part_000.pth`、`partitions/part_000.pth` 是否完整生成。
4. 再跑 `examples/bench_chunk_native.py --mode train` 验证 CTDG sampled edge train。
5. 最后再接入 FlareDTDG STGraph 和 MemShare mailbox 两条训练入口。
