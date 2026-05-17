# Chunk Backend 方法设计

## 1. 设计定位

`backends/chunk` 定位为面向 DTDG/CTDG 训练执行的低开销数据管理与规划层。

它不强行把 DTDG 的完整子图数据和 CTDG 的采样 MFG 数据合并成同一种内部表示，而是在 placement、通信 plan、profile、migration 等外层 API 上进行对齐，同时保留每种训练模态最高效的 payload 格式。

核心原则：

- 通过 `PartitionData` 只保存一份 canonical graph artifact。
- 从同一份图数据派生 DTDG 和 CTDG 视图，避免复制完整图。
- CTDG 外层输入保持为 event stream，sampler 内部使用 temporal index，采样输出保持为 MFGs/native blocks。
- DTDG 输入保持为完整 snapshot/window 子图 payload。
- DTDG/CTDG 尽量对齐 `comm_plan` 和 envelope 外层 API。
- 通过 tensor view、range、offset、id、连续 tensor bundle 降低数据实例化开销。
- chunk 迁移仅发生在跳参训练 trial 之间，不在训练过程中动态迁移。
- 当前改动限制在 `backends/chunk` 内部；需要复用 DTDG/CTDG 的概念时，在 chunk 目录下复制或平行实现接口，不修改 `backends/dtdg` 或 `backends/ctdg`。

## 2. 当前执行流程

### 2.1 Prepare 流程

```text
raw events / edge file
  -> RawTemporalEvents
  -> snapshot dataset，用于 split/time_ptr 元数据
  -> prepare_chunks(...)
  -> 生成 node partition
  -> 选择 hot nodes
  -> 构建 chunk assignment
  -> 统计 load
  -> chunk owner rebalance
  -> 可选构建 memory route
  -> 写出 PartitionData artifact
  -> 写出 metadata/manifest
```

主要函数：

- `ChunkPreprocessor.prepare_raw()`
- `ChunkPreprocessor.build_partitions()`
- `ChunkPreprocessor.build_runtime_artifacts()`
- `backends/chunk/prepare/pipeline.py` 中的 `prepare_chunks()`
- `PartitionData.from_edge_events()`

主要 artifact：

```text
artifact_root/
  meta/artifacts.json
  meta.json
  partitions/manifest.json
  partitions/rebalance_manifest.json
  chunk_assignment.pth
  node_partition.pt
  node_owner.pt
  hot_node_ids.pt
  replica_mask.pt
  time_ptr.pt
  part_000.pth
  routes/manifest.json
  snapshots/manifest.json
  clusters/part_000/cluster_manifest.json
```

### 2.2 Runtime 流程

```text
SchedulerSession.build_runtime()
  -> ChunkRuntimeLoader.from_prepared_artifacts(...)
  -> 加载 PartitionData
  -> 加载可选 memory/spatial routes
  -> 基于 PartitionData 构建 ChunkGraphStore
  -> 构建 task adapter
  -> 构建 CommPipeline
  -> 暴露 BatchData iterator 和 ExecutionUnit iterator
```

现有兼容路径：

```text
run_epoch()
  -> graph_runtime.iter_train()/iter_eval()
  -> BatchData
  -> graph_runtime.run_train_step()/run_eval_step()
  -> run_batch(...)
```

新增对齐 API 路径：

```text
iter_train_envelopes()/iter_eval_envelopes()/iter_predict_envelopes()
  -> ExecutionUnit
  -> envelope.payload 保持当前模态原生格式
  -> envelope.comm_plan 使用对齐后的 Fetch/Propagation/StateSync plan wrapper
```

## 3. Canonical Data Layer

### 3.1 PartitionData

文件：`backends/chunk/data/partition.py`

`PartitionData` 是 chunk-backed DTDG/CTDG 执行的 canonical graph artifact。

它支持可逆转换：

```python
PartitionData.from_edge_index(...)
PartitionData.from_edge_events(...)
PartitionData.to_edge_index(...)
PartitionData.edge_events(...)
```

用途：

- 只保存一份完整图数据。
- 支持构建 DTDG snapshot/full-subgraph view。
- 支持构建 CTDG event 和 temporal sampler 输入。
- 保留稳定的 `edge_id` 引用。
- 通过 `node_to_chunk` 和 `dst_chunk` 附加 chunk 元数据。

关键内部布局：

- `TensorData` 使用连续 `data` 加 `ptr` 打包可变长度 per-snapshot tensors。
- 边按 destination chunk 和 destination node 排序，以改善 locality。
- `edge_ids` 保留原始边的 canonical 引用。
- 当存在时间戳时，`edge_data["timestamps"]` 存储事件时间。

### 3.2 ChunkGraphStore

文件：`backends/chunk/data/graph_store.py`

```python
@dataclass(slots=True)
class ChunkGraphStore:
    part: PartitionData
    placement: ChunkPlacement
    _temporal_index_cache: Optional[TemporalIndexView] = None
```

职责：

- 将 `PartitionData` 视为唯一完整图 artifact。
- 提供轻量 DTDG/CTDG views。
- 为 sampler/native path 构建并缓存 temporal adjacency index。
- DTDG/CTDG 模态切换时不复制完整图。

#### `ChunkGraphStore.from_partition_data(part, placement=None)`

从已有 `PartitionData` 构建 graph store。

如果没有传入 placement，则创建默认 placement：

```text
node_to_chunk = part.node_to_chunk or zeros
node_owner = zeros
node_master = node_owner
replica_mask = zeros
placement_version = 0
```

#### `ChunkGraphStore.temporal_index_view(sort_by_time=True)`

返回供 sampler/native path 使用的 `TemporalIndexView`。

输出：

```python
TemporalIndexView(
    indptr,
    indices,
    edge_ids,
    timestamps,
    placement,
)
```

算法：

1. 通过 `part.edge_events(global_ids=True, sort_by_timestamp=False)` 提取 canonical events。
2. 根据 placement 和边端点推断 node count。
3. 如果 `sort_by_time=True` 且存在 timestamps：
   - 先按 timestamp 做 stable sort；
   - 再在 timestamp-sorted order 上按 destination 做 stable sort；
   - 这样保证每个 destination group 内部尽量保持时间顺序。
4. 否则只按 destination 做 stable sort。
5. 使用 `torch.bincount(sorted_dst)` 和 prefix sum 构建 CSC `indptr`。
6. 将 source nodes 存入 `indices`。
7. 将稳定的 canonical edge ids 存入 `edge_ids`。
8. 输出 tensor 保持 contiguous。
9. 当 `sort_by_time=True` 时缓存结果。

复杂度：

- 排序：每个 graph store 一次 `O(E log E)`。
- CSC pointer 构建：`O(E + N)`。
- 后续缓存访问：`O(1)`。

#### `ChunkGraphStore.ctdg_input_view(batch_id, event_start, event_end)`

构建轻量 CTDG event-batch view。

输出：

```python
EventView(
    batch_id,
    event_start,
    event_end,
    root_nodes,
    root_ts,
    temporal_index,
    placement_version,
)
```

算法：

1. 通过 `part.edge_events(sort_by_timestamp=True)` 获取按时间排序的 events。
2. 切片 `[event_start:event_end]`。
3. 计算该 event slice 的 root nodes，即 unique `src ∪ dst`。
4. 复用缓存的 CSC view。

返回结果只是 view，不 materialize sampled MFGs。

#### `ChunkGraphStore.dtdg_input_view(snapshot_id, window_start, window_end, payload=None)`

构建 DTDG snapshot/window view。

输出：

```python
SnapshotView(
    snapshot_id,
    window_start,
    window_end,
    payload,
    node_ids,
    edge_ids,
    placement_version,
)
```

算法：

1. 通过 `part.to_edge_index(snapshot_index=snapshot_id)` 恢复 snapshot edge index。
2. 收集 endpoint node ids 的 unique 集合。
3. 读取稳定的 snapshot edge ids。
4. `payload` 保持当前模态原生格式，例如 DGLBlock/STGraphBlob。

## 4. 统一外层 API

文件：`backends/chunk/data/plans.py`

下面这些 dataclass 用于对齐 DTDG/CTDG 的外层接口，但不强制底层 payload 内部表示一致。

### 4.1 PlacementView

```python
@dataclass(slots=True)
class PlacementView:
    placement_version: int
    node_to_chunk: Tensor
    node_owner: Tensor
    node_master: Tensor
    replica_mask: Tensor
```

用途：

- DTDG/CTDG views 共享的节点 placement table。
- 训练 plan 携带 `placement_version`，用于避免 stale plan。

### 4.2 TemporalIndexView

```python
@dataclass(slots=True)
class TemporalIndexView:
    indptr: Tensor
    indices: Tensor
    edge_ids: Tensor
    timestamps: Optional[Tensor]
    placement: PlacementView
```

用途：

- CTDG full-graph input。
- Temporal sampler 直接消费 CSC/temporal CSC。
- `edge_ids` 回指 canonical `PartitionData`。

### 4.3 SnapshotView

```python
@dataclass(slots=True)
class SnapshotView:
    snapshot_id: int
    window_start: int
    window_end: int
    payload: Any
    node_ids: Optional[Tensor]
    edge_ids: Optional[Tensor]
    placement_version: int
```

用途：

- 包装 DTDG 原生完整子图/window payload。
- 不强制转换成 CTDG MFG 格式。

### 4.4 EventView

```python
@dataclass(slots=True)
class EventView:
    batch_id: int
    event_start: int
    event_end: int
    root_nodes: Tensor
    root_ts: Tensor
    temporal_index: TemporalIndexView
    placement_version: int
```

用途：

- CTDG sampler 输入。
- 整图保持 CSC，采样结果保持 MFGs。

### 4.5 CTDGSampleResult

```python
@dataclass(slots=True)
class CTDGSampleResult:
    mfgs: list[Any]
    input_nodes: Tensor
    output_nodes: Tensor
    edge_ids: Tensor
    node_ts: Optional[Tensor] = None
    edge_ts: Optional[Tensor] = None
    memory_node_ids: Optional[Tensor] = None
    remote_node_ids: Optional[Tensor] = None
```

用途：

- CTDG sampler output 的标准 wrapper。
- `mfgs` 保持 backend-native，以保证性能。

### 4.6 BlockPlan

```python
@dataclass(slots=True)
class BlockPlan:
    mode: Literal["dtdg", "ctdg"]
    block_id: int
    placement_version: int
    time_range: Optional[tuple[float, float]] = None
    snapshot_range: Optional[tuple[int, int]] = None
    event_range: Optional[tuple[int, int]] = None
    root_nodes: Optional[Tensor] = None
    edge_ids: Optional[Tensor] = None
    load_hint: dict[str, float] = field(default_factory=dict)
```

用途：

- 统一调度单元。
- DTDG 使用 `snapshot_range`。
- CTDG 使用 `event_range`。

### 4.7 PlanBundle

```python
@dataclass(slots=True)
class PlanBundle:
    fetch: Optional[FetchPlan] = None
    propagation: Optional[PropagationPlan] = None
    state_sync: Optional[StateSyncPlan] = None
```

用途：

- 对齐三类通信 plan。
- 每类 plan 都可以不存在。

### 4.8 FetchPlan

```python
@dataclass(slots=True)
class FetchPlan:
    block_id: int
    placement_version: int
    feature_node_ids: Tensor
    feature_owners: Tensor
    memory_node_ids: Optional[Tensor] = None
    memory_owners: Optional[Tensor] = None
    cache_policy: str = "none"
```

用途：

- Batch prepare 阶段通信。
- 覆盖 feature fetch 和 remote memory fetch。
- 不参与 autograd。

### 4.9 PropagationPlan

```python
@dataclass(slots=True)
class PropagationPlan:
    block_id: int
    placement_version: int
    layer_routes: list[Any]
    autograd_enabled: bool = True
```

用途：

- Model-layer message/activation propagation。
- 参与 autograd。
- 可以使用 `ChunkPropagationRoute` 或当前模态原生 route。

### 4.10 StateSyncPlan

```python
@dataclass(slots=True)
class StateSyncPlan:
    block_id: int
    placement_version: int
    update_node_ids: Tensor
    update_owners: Tensor
    replica_node_ids: Optional[Tensor] = None
    replica_owners: Optional[Tensor] = None
    sync_policy: str = "owner_write"
```

用途：

- Batch commit 阶段 state/memory/cache update。
- 覆盖 CTDG memory/mailbox update 和 hot-replica sync。
- 不属于 model-layer autograd。

### 4.11 ExecutionUnit

```python
@dataclass(slots=True)
class ExecutionUnit:
    mode: Literal["dtdg", "ctdg"]
    block_id: int
    placement_version: int
    payload: Any
    comm_plan: PlanBundle = field(default_factory=PlanBundle)
    profile_hint: dict[str, float] = field(default_factory=dict)
```

用途：

- 外层 API 对齐。
- `payload` 保持原生格式：
  - DTDG：DGLBlock/STGraphBlob/full-subgraph payload。
  - CTDG：MFG/sample result payload。
  - 当前兼容路径：`BatchData`。
- `GraphBatchEnvelope` 仅作为旧命名兼容别名保留，新增代码使用 `ExecutionUnit`。

### 4.12 ProfileRecord

```python
@dataclass(slots=True)
class ProfileRecord:
    mode: Literal["dtdg", "ctdg"]
    block_id: int
    placement_version: int
    sample_ms: float = 0.0
    fetch_ms: float = 0.0
    propagation_ms: float = 0.0
    state_sync_ms: float = 0.0
    compute_ms: float = 0.0
    num_nodes: int = 0
    num_edges: int = 0
    num_remote_nodes: int = 0
    num_memory_nodes: int = 0
    peak_mem_bytes: int = 0
```

用途：

- 为跳参训练时的 replanning 提供统一 profile record。
- Planner 不需要理解底层 payload 是 DGLBlock 还是 MFG。

### 4.13 ChunkPlacement 和 PlacementDelta

```python
@dataclass(slots=True)
class ChunkPlacement:
    placement_version: int
    node_to_chunk: Tensor
    node_owner: Tensor
    node_master: Tensor
    replica_mask: Tensor
```

```python
@dataclass(slots=True)
class PlacementDelta:
    old_version: int
    new_version: int
    moved_chunks: Tensor
    affected_nodes: Tensor
    affected_blocks: Tensor
    affected_routes: Tensor
```

用途：

- 一个 training run 内 placement 固定。
- 迁移只发生在跳参 trial 之间。
- Delta 只描述部分迁移。

## 5. 通信类型

对齐后的 API 将通信拆成三类。

### 5.1 Fetch 通信

Plan 类型：`FetchPlan`

阶段：

```text
batch prepare / prefetch
```

数据：

- node features
- 如有需要，edge features
- CTDG remote memory

性质：

- 只读。
- 通常不参与 autograd。
- 可与上一个 batch 的 compute 重叠。

### 5.2 Propagation 通信

Plan 类型：`PropagationPlan`

阶段：

```text
model forward/backward 内部
```

数据：

- layer activation
- GNN messages
- boundary node embeddings

性质：

- autograd-aware。
- 属于 model/MFG/layer execution。
- 不应该在 dataloader 中提前完成。

### 5.3 State Synchronization 通信

Plan 类型：`StateSyncPlan`

阶段：

```text
batch commit / post compute
```

数据：

- memory delta
- mailbox update
- owner writeback
- hot replica sync

性质：

- update/write 操作。
- 通常不属于 model-layer autograd。
- 依赖 placement/master 语义。

## 6. ChunkPropagationRoute

文件：`backends/chunk/data/propagation_route.py`

`ChunkPropagationRoute` 是 DTDG autograd propagation route primitive 在 chunk 目录下的复制版本。

接口：

```python
@dataclass(slots=True)
class ChunkPropagationRoute:
    send_sizes: list[int]
    recv_sizes: list[int]
    send_index: Optional[Tensor] = None
    group: Optional[dist.ProcessGroup] = None

    def forward(self, x: Tensor, reverse: bool = False, group=None) -> Tensor:
        ...

    async def async_forward(self, x: Tensor, reverse: bool = False, group=None) -> Tensor:
        ...
```

算法：

1. 如果 `send_index is None` 或 distributed 未初始化，直接返回 `x`。
2. 使用 `x[send_index].contiguous()` gather 本地 send buffer。
3. 根据 send/recv sizes 调用 `dist.all_to_all_single`。
4. 返回收到的远程 activation rows。
5. backward 阶段：
   - 反向执行 all-to-all；
   - 使用 `index_add_` 将收到的梯度 scatter 回原始本地 tensor rows。

说明：

- 该 route 只用于 model-layer propagation。
- 它与 fetch/state-sync plans 分离。
- 放在 `backends/chunk` 内，避免修改 `backends/dtdg`。

## 7. ChunkRuntimeLoader API 对齐

文件：`backends/chunk/runtime/loader.py`

### 7.1 现有兼容 Iterator

这些接口保持不变，用于保证现有代码可继续运行：

```python
def iter_train(self, split: str = "train") -> Iterator[BatchData]
def iter_eval(self, split: str = "val") -> Iterator[BatchData]
def iter_predict(self, split: str = "test") -> Iterator[BatchData]
```

它们返回当前 `BatchData`，保持原 train/predict 链路可用。

### 7.2 新增 ExecutionUnit Iterator

```python
def iter_train_units(self, split: str = "train") -> Iterator[ExecutionUnit]
def iter_eval_units(self, split: str = "val") -> Iterator[ExecutionUnit]
def iter_predict_units(self, split: str = "test") -> Iterator[ExecutionUnit]
```

算法：

1. CTDG/link prediction 分支使用 `MemShareEventEngine.iter_units()`。
2. `payload` 为 `EventView`，内部持有共享 CSC 和连续 root tensors。
3. DTDG 或兼容分支退回 `_iter_split_with_index(split)`，payload 为 `BatchData`。
4. 如果 route artifact 存在，则附加 `PlanBundle`。
5. 不 materialize 新的 graph payload。

### 7.3 `_make_envelope(batch, block_id)`

输出：

```python
ExecutionUnit(
    mode=...,
    block_id=block_id,
    placement_version=graph_store.placement_version,
    payload=batch,
    comm_plan=_make_comm_plan(block_id),
    profile_hint={...},
)
```

当前 mode 推断是保守策略：

- link/edge 类 task 被视作 CTDG-like。
- node/snapshot 类 task 被视作 DTDG-like。

该判断不改变 payload 内部结构。

### 7.4 `_make_comm_plan(block_id)`

从当前已有 route 对象构建对齐后的通信 plan。

如果存在 `spatial_routes[block_id]`：

```text
SpatialRouteData -> FetchPlan
```

算法：

1. 使用 `route.recv_node_ids` 作为 `feature_node_ids`。
2. 通过 `recv_ptr` 和 `repeat_interleave` 推导 `feature_owners`。
3. 构建 `FetchPlan`。

如果存在 `mem_routes[block_id]`：

```text
MemoryRouteData -> StateSyncPlan
```

算法：

1. 使用 `route.unique_nodes` 作为 `update_node_ids`。
2. 通过 `send_ptr` 和 `repeat_interleave` 推导 `update_owners`。
3. 如果存在 `replica_idx`，设置 `replica_node_ids`。
4. 构建 `StateSyncPlan`。

`PropagationPlan` 当前保持 `None`，等后续 DTDG/CTDG model-layer route 接入。

## 8. 函数内部算法说明

### 8.1 `prepare_chunks(...)`

文件：`backends/chunk/prepare/pipeline.py`

主要算法：

1. 校验 partition 数量。
2. 如果没有传入 `node_partition`：
   - 计算节点 degree；
   - 根据 `hot_topk` 或 `hot_ratio` 选择 hot nodes；
   - 使用 `metis` 或 `mem_share` 策略划分 cold nodes；
   - 根据邻居关联度和负载 penalty 为 hot node 选择 master；
   - 为 hot nodes 创建 `replica_mask`。
3. 使用 `build_chunk_assignment()` 构建 chunk assignment。
4. 统计 load：
   - 如果提供 `time_ptr`，执行 windowed load stats；
   - 否则执行 single-pass load stats。
5. 执行 chunk owner rebalance。
6. 将 hot-node owner 恢复为基于 affinity 选择的 master。
7. 可选构建 memory routes。
8. 可选构建 spatial routes。
9. 返回 prepare artifacts。

性能说明：

- degree 和 load counting 尽量使用 vectorized scatter/bincount。
- Python loop 只保留在 coarse level，例如 time windows 和 chunks。
- 当前 fallback 路径避免直接调用 DGL METIS，因为 native failure 无法被 Python 稳定捕获。

### 8.2 `build_chunk_assignment(node_partition, num_chunks_per_partition)`

文件：`backends/chunk/prepare/chunk_assignment.py`

算法：

1. 对每个 partition，列出属于该 partition 的节点。
2. 根据节点 rank 分配 local chunk id：

```text
local_chunk = rank * num_chunks_per_partition // num_partition_nodes
```

3. 计算 global chunk id：

```text
global_chunk = partition_id * num_chunks_per_partition + local_chunk
```

4. 按 chunk id 对节点排序。
5. 通过 bincount 和 slicing 构建 `chunk_to_nodes`。
6. 初始化 `chunk_to_initial_partition` 和 `chunk_to_owner_partition`。

### 8.3 `compute_chunk_load_stats(...)`

文件：`backends/chunk/prepare/load_stats.py`

算法：

1. 将每条边的 destination 映射到 `dst_chunk`。
2. 将 destination chunk 映射到 owner partition。
3. 使用 `scatter_add_` 统计每个 chunk 的 edge count。
4. 统计 remote edges，即 `src_partition != dst_owner` 的边。
5. 通过编码 pair 统计每个 chunk 的 unique active source nodes：

```text
combined = dst_chunk * (num_nodes + 1) + src_node
```

6. 使用相同方式统计 unique remote source nodes。
7. 构建 `ChunkLoadStats` 并计算 weighted total load。

### 8.4 `rebalance_chunks(...)`

文件：`backends/chunk/prepare/rebalancer.py`

算法：

1. 按 owner partition 汇总 chunk load。
2. 按 load 降序排列 chunks。
3. 当 imbalance 超过阈值时：
   - 选择一个重 chunk；
   - 仅当迁移到当前最轻 partition 能改善 imbalance 时才执行迁移；
   - 将迁移记录到 manifest。
4. 根据 chunk owner table 推导最终 `node_owner`。

### 8.5 `PartitionData.from_edge_events(...)`

文件：`backends/chunk/data/partition.py`

算法：

1. 可选地对 event tensors 做 slice。
2. 委托给 `from_edge_index()`。
3. `from_edge_index()` 构建紧凑 partition tensors：
   - 按 destination chunk 和 destination 排序；
   - 构建 local source/destination id spaces；
   - 构建 edge CSR pointer；
   - 将 timestamps 存入 `edge_data`。

### 8.6 `PartitionData.edge_events(...)`

算法：

1. 将选中的 snapshots 转回 global edge index。
2. 如果存在 timestamp 和 edge id，则一并收集。
3. 可选地按 timestamp 排序返回 events。
4. 返回 `(src, dst, timestamps, edge_ids)`。

这是 canonical `PartitionData` 到 CTDG event sequence 的桥接接口。

## 9. 当前限制

当前实现已完成 API 对齐、兼容路径保留，以及 chunk-local MemShare event 执行入口。

已知限制：

- 当前兼容路径下，`iter_train()/iter_eval()/iter_predict()` 仍返回现有 `BatchData`，保证 train/predict 不被破坏。
- 新 event 路径通过 `iter_train_units()/iter_eval_units()/iter_predict_units()` 返回 `ExecutionUnit`，payload 为连续 `EventView`。
- `PropagationPlan` 尚未由 model-layer routes 填充。
- `MemShareEventEngine` 已在 `backends/chunk/runtime/event_engine.py` 接入 native sampler，输出 `CTDGSampleResult`；无 timestamp 的 mock 图会拒绝 native sample。
- 一些 route artifact 仍是 object/list 格式；后续应改为 packed tensor bundle。
- `PartitionData` 当前支持转换和 CSC view 构建，但 streaming prepare 和 mmap-backed storage 仍是后续工作。
- `PlacementDelta` 已定义，但 migration executor 尚未实现。

## 10. 建议的下一步实现

1. 将训练入口从 `BatchData` 兼容路径逐步切换到 `ExecutionUnit`，CTDG 分支直接消费 `EventView + CTDGSampleResult`。
2. 在需要 model-layer 通信的位置，用 `ChunkPropagationRoute` 或当前模态原生 route 填充 `PropagationPlan`。
3. 将 object-per-slice route files 替换为 packed tensor route bundles。
4. 实现仅在跳参 trial 之间执行的 `PlacementDelta` migration。
5. 增加 `ProfileRecord` 采集，用于 block-level replanning。
6. 将 mock/小数据测试扩展为带 timestamp 的真实数据 smoke test，覆盖 native sample。
