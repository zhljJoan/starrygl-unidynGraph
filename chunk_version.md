# Backends/Chunk 最新链路说明

## 1. 整体定位

`backends/chunk` 定位为面向训练的统一数据管理与调度层。它不替代 DTDG 或 CTDG 的模型计算方式，而是在训练前完成时空切分、负载统计、chunk 分配、热点复制和通信计划构建，运行时再根据训练模态派生对应的数据视图。

核心目标是：只保存一份完整图数据，避免 DTDG/CTDG 各自维护数据副本；同时不强行统一两类训练模态的内部格式，保留各自高性能链路。

## 2. Prepare 链路

Prepare 阶段负责从原始动态图数据生成 chunk 运行所需 artifact。

主要流程：

1. 读取原始 event/edge 文件，生成按时间排序的事件序列。
2. 根据训练配置执行时间切片，得到 snapshot/window 范围。
3. 执行空间切片，生成节点到 chunk 的初始分配。
4. 统计每个时空块的负载向量，包括边数量、活跃节点、远程访问量和热点关联信息。
5. 基于负载均衡策略分配 chunk owner。
6. 识别热点节点，热点数据作为共享 chunk 在各机器复制。
7. 为后续训练生成必要的通信路由和 metadata。
8. 将统一图数据、chunk assignment、placement、route 和 manifest 落盘。

当前原始文本读取已从 Python 逐行解析调整为 pandas mmap 读取，后续仍需要继续优化全量 prepare 的 packed tensor artifact 和分块处理。

## 3. 统一数据层

Chunk 链路的基础是一份 canonical graph artifact。该 artifact 保存完整图事件、时间戳、节点划分、chunk 分配和必要的索引信息。

该层的设计原则：

- 图数据只保存一份，避免 DTDG/CTDG 双份拷贝。
- DTDG 和 CTDG 从同一份数据派生不同训练视图。
- 派生视图尽量是轻量 view、offset、range 或连续 tensor，不在 dataloader 中重复构造完整图。
- 数据结构优先保证访存连续性，方便后续 native/C++ 执行路径接入。

## 4. DTDG 视图定位

DTDG 的训练对象是时间切片后的完整子图或历史快照窗口。

因此 DTDG 应从统一图数据中派生 snapshot/window 级 CSC/CSR 视图，用于完整子图上的邻接遍历、聚合、历史快照衰减采样和跨分区消息传播。

DTDG 不应该被强行转换成 CTDG 的 event batch 或 MFG 格式。Chunk 只负责提供 snapshot 范围、placement 信息、通信计划和负载统计，具体快照训练仍保留 DTDG 的高效子图计算方式。

## 5. CTDG 视图定位

CTDG 的训练对象是连续事件流和事件 batch。

CTDG 从统一图数据中派生 event stream 视图，并在采样阶段使用 temporal sampling index。MemShare/native sampler 内部可以维护 temporal adjacency index，但这属于采样索引，不是 CTDG 外层训练输入格式。

CTDG 的训练输入应保持为采样后的 MFG/native block。这样可以避免把 CTDG 强行转换成完整 CSC 子图，减少 dataloader 转换和额外拷贝。

当前 CTDG 方向以 MemShare native 作为最终路径，后续重点是将 dedup、采样结果整理和 MFG 构建继续下沉到 C++ native。

## 6. Runtime 链路

Runtime 阶段负责加载 prepare artifact，并向训练循环提供对应模态的数据单元。

当前链路包含两条路径：

1. 兼容路径：继续输出现有 BatchData，保证 train/predict 链路可以稳定跑通。
2. 新统一路径：输出 ExecutionUnit，payload 根据训练模态保持原生格式。

ExecutionUnit 只作为外层调度单元，负责携带 mode、block id、placement version、payload、通信计划和 profile hint。它不强制 payload 使用同一种内部结构。

## 7. 通信设计

Chunk 链路中通信分为四类：

1. Feature fetch / remote memory fetch：发生在 batch prepare 阶段，用于获取远程特征或远程 memory。
2. Model-layer message propagation：发生在模型前向和反向传播中，需要保留 autograd 支持。
3. Memory/cache state sync：发生在 batch commit 阶段，用于同步 memory、mailbox 或 cache 状态。
4. Gradient sync：属于分布式训练框架层，由 DDP/torch distributed 管理。

前三类通信在 chunk 外层 plan 上统一描述，但执行位置不同，不能混成一个通信阶段。尤其模型层消息传播必须穿插在模型计算内部。

## 8. Chunk 迁移策略

当前设计不在训练过程中动态迁移 chunk。

Chunk 迁移只发生在跳参训练 trial 之间。训练过程中采集 profile 信息，下一次任务或下一组参数启动前，根据新的负载统计和 profile 结果调整少量 chunk 分配。

这样可以避免训练中迁移带来的状态一致性、memory ownership 和通信路由失效问题，同时保留跨任务自适应分配能力。

## 9. 采样策略

采样策略按训练模态区分：

- DTDG 保留历史快照衰减采样和完整子图采样逻辑。
- CTDG 使用 event batch 驱动，并通过 MemShare/native temporal sampler 生成 MFG/native block。
- Chunk 负责提供时空块、placement、热点信息和负载统计，使采样可以感知数据局部性和跨分区代价。

后续可以在现有采样基础上加入时空块感知策略，例如优先本地邻居、热点副本命中、跨分区访问代价约束和基于 profile 的自适应 fanout。

## 10. 当前验证状态

当前已完成以下 smoke 验证：

- 使用真实 `soc-bitcoin` 截断数据跑通 chunk prepare。
- 跑通 snapshot node regression 的 train/predict 链路。
- 跑通 temporal link prediction 的 train/predict 链路。
- 验证 CTDG event view 可以生成 ExecutionUnit。
- 验证 MemShare native sampler 可以基于真实 artifact 返回 CTDGSampleResult。
- 原始文本读取已切换为 pandas mmap 读取，50k 真实边 prepare 约 0.1s。

全量超大数据仍需要继续优化 prepare 阶段，包括分块读取、packed tensor artifact、route 文件合并和减少 Python object/list。

## 11. 下一阶段重点

1. 将 prepare artifact 从多 object 文件逐步改为 packed tensor bundle。
2. 优化全量数据的分块读取和落盘，减少一次性中间对象。
3. 将 CTDG dedup、采样结果整理和 MFG 构建下沉到 C++ native。
4. 将训练入口从 BatchData 兼容路径逐步切换到 ExecutionUnit 路径。
5. 补齐 model-layer propagation plan，并保证正向/反向传播通信正确。
6. 增加 profile 采集，用于跨 trial 的 chunk 重分配。

## 12. 最新执行流程总览

### 12.1 CLI / Session 入口

1. `python -m starry_unigraph --phase prepare|train|predict`
2. 读取配置，确认 `data.graph_mode = chunk`。
3. `SchedulerSession.prepare_data()` 在 prepare 阶段选择 `ChunkPreprocessor`。
4. `SchedulerSession.build_runtime()` 在 train/predict 阶段选择 `ChunkRuntimeLoader`。
5. `SchedulerSession.run_epoch()` 驱动训练或验证。
6. `SchedulerSession.predict()` 驱动预测。

这一层负责模式选择和训练循环调度，不直接处理 chunk 内部数据结构。

### 12.2 Prepare 执行顺序

1. `ChunkPreprocessor.prepare_raw()`
   读取原始动态图数据，生成统一事件序列，并根据配置生成时间切片后的 snapshot dataset。

2. `load_raw_temporal_events()`
   负责读取原始 edge/event 文件。当前 `.edges` 文本路径使用 pandas mmap 读取，避免 Python 逐行解析。

3. `build_snapshot_dataset_from_events()`
   根据事件序列和 `train.snaps` / `chunk.time_slices` 生成时间窗口，用于构造 `time_ptr` 和 snapshot 元数据。

4. `ChunkPreprocessor.build_partitions()`
   调用 chunk prepare 主流程，完成节点划分、热点识别、chunk assignment、负载统计和重分配。

5. `prepare_chunks()`
   Chunk prepare 的核心函数。输入全量事件边、时间切片、分区数和 chunk 配置，输出 `PrepareArtifacts`。

6. `build_node_partition()`
   当外部没有提供 node partition 时生成节点初始分区。支持 `metis` 和 `mem_share` 两种策略，并处理热点节点 master 分配。

7. `build_chunk_assignment()`
   将 node partition 进一步切成 chunk，生成 `node_to_chunk`、`chunk_to_nodes`、`chunk_to_initial_partition`、`chunk_to_owner_partition`。

8. `compute_chunk_load_stats()` / `compute_chunk_load_stats_from_windows()`
   根据时空窗口统计每个 chunk 的负载，包括边数量、活跃节点、远程边和远程节点。

9. `rebalance_chunks()`
   根据 chunk load 调整 `chunk_to_owner_partition`，输出新的 node owner 和迁移 manifest。

10. `build_memory_route_phase1()` / `assign_memory_route_ptrs()`
    可选构建 memory/cache 状态同步路由。

11. `build_spatial_routes()`
    可选构建 feature fetch 或远程节点访问路由。

12. `ChunkPreprocessor.build_runtime_artifacts()`
    将 `PartitionData`、`chunk_assignment.pth`、`node_owner.pt`、`time_ptr.pt`、routes 和 manifest 写入 artifact 目录。

### 12.3 Prepare 核心类

- `RawTemporalEvents`
  原始事件序列，保存 `src`、`dst`、`ts`、`weight`、`edge_feat`、`num_nodes` 和 `num_edges`。

- `PrepareArtifacts`
  prepare 主流程输出，包含 `ChunkAssignment`、`node_owner`、负载统计、热点信息、route 和 `time_ptr`。

- `ChunkAssignment`
  空间布局核心结构，保存 node、chunk、partition 三者之间的映射关系。

- `ChunkLoadStats`
  单个 chunk 的负载统计，用于重分配和后续 profile 对齐。

- `ChunkReassignmentManifest`
  记录 chunk owner 调整结果，后续跨 trial 迁移会基于它扩展。

- `PartitionData`
  落盘的 canonical graph artifact。运行时 DTDG/CTDG 都从这里派生视图。

## 13. Runtime 执行顺序

### 13.1 Runtime 构建

1. `SchedulerSession.build_runtime()`
   读取 prepare 阶段的 metadata，并校验 graph mode 和 world size。

2. `ChunkRuntimeLoader.from_prepared_artifacts()`
   加载当前 rank 的 `part_{rank}.pth`、memory routes、spatial routes、`meta.json` 和 manifest。

3. `ChunkGraphStore.from_partition_data()`
   基于 `PartitionData` 构建统一图存储对象，负责后续派生 DTDG/CTDG 视图。

4. `get_task_adapter()`
   根据任务类型选择 batch 构建和 loss/metric 逻辑。

5. `MemShareEventEngine.from_config()`
   为 CTDG/event 路径准备 MemShare native sampler 的 Python 侧执行入口。

6. `CommPipeline`
   初始化通信流水线对象，当前主要作为后续异步通信接入点。

### 13.2 兼容训练路径

当前 train/predict 默认仍走兼容路径，执行顺序如下：

1. `SchedulerSession.run_epoch(split="train")`
2. `ChunkRuntimeLoader.iter_train()`
3. `ChunkRuntimeLoader._iter_split_with_index()`
4. `task_adapter.build_batch()`
5. 产出 `BatchData`
6. `ChunkRuntimeLoader.run_train_step()`
7. `run_batch()`
8. `task_adapter.compute_loss()` 和 `compute_metrics()`

predict 路径类似：

1. `SchedulerSession.predict(split="test")`
2. `ChunkRuntimeLoader.iter_predict()`
3. `task_adapter.build_batch()`
4. `ChunkRuntimeLoader.run_predict_step()`
5. 返回 predictions 和 targets。

这条路径用于保证当前链路稳定可跑，但不是最终高性能 CTDG 路径。

### 13.3 ExecutionUnit 新路径

ExecutionUnit 是后续统一高性能路径的外层执行单元。

CTDG/link prediction 分支当前执行顺序：

1. `ChunkRuntimeLoader.iter_train_units()`
2. `ChunkRuntimeLoader._iter_units()`
3. `MemShareEventEngine.iter_units()`
4. `ChunkGraphStore.event_range_for_snapshot()`
5. `ChunkGraphStore.ctdg_input_view()`
6. 构造 `EventView`
7. `MemShareEventEngine.make_unit()`
8. 产出 `ExecutionUnit(mode="ctdg", payload=EventView)`
9. `MemShareEventEngine.sample()`
10. 调用 MemShare native sampler
11. 返回 `CTDGSampleResult`

DTDG 或非 link prediction 分支当前仍回退到兼容路径：

1. `ChunkRuntimeLoader.iter_train_units()`
2. `_iter_envelopes()`
3. `_iter_split_with_index()`
4. `task_adapter.build_batch()`
5. `ExecutionUnit(payload=BatchData)`

后续目标是让 DTDG 分支输出 `SnapshotView` 或完整 snapshot/window 原生 payload，而不是继续包装 `BatchData`。

## 14. Runtime 核心类

- `ChunkRuntimeLoader`
  Runtime 主入口，负责加载 artifact、构建 task adapter、构建 graph store、暴露 batch iterator 和 execution unit iterator。

- `ChunkGraphStore`
  运行时统一图存储。负责从 `PartitionData` 派生 `TemporalEventTable`、`EventView`、`SnapshotView` 和内部索引视图。

- `TemporalEventTable`
  CTDG event path 的连续事件表，包含 `src`、`dst`、`ts`、`edge_ids` 和 `snapshot_event_ptr`。

- `EventView`
  CTDG event batch 视图。保存 event range、root nodes、root timestamps 和采样所需索引引用。

- `SnapshotView`
  DTDG snapshot/window 视图。用于表达完整子图或快照窗口 payload。

- `ExecutionUnit`
  新统一执行单元。只对齐外层调度字段，不强制内部 payload 格式统一。

- `PlanBundle`
  三类非梯度通信计划的外层包装，包括 fetch、propagation 和 state sync。

- `MemShareEventEngine`
  CTDG event 路径的执行引擎，负责从 `EventView` 调用 MemShare native sampler 并返回 `CTDGSampleResult`。

- `CTDGSampleResult`
  CTDG 采样结果，保持 MFG/native block，不转成 DTDG 完整子图格式。

## 15. 当前 Code Review 结论

### 15.1 已跑通部分

- Chunk prepare 可以从真实 edge 文件生成 artifact。
- train/predict 兼容路径可以跑通。
- CTDG event path 可以生成 `ExecutionUnit + EventView`。
- MemShare native sampler 可以基于真实 artifact 返回 `CTDGSampleResult`。
- pandas mmap 读取已经替代 Python 逐行解析。

### 15.2 当前主要风险

1. Runtime 默认 placement 没有加载 `node_owner.pt` 和 `replica_mask.pt`。
   目前 `ChunkGraphStore.from_partition_data()` 默认把 `node_owner` 置零，这会影响多机 owner、route 和 native sampler 的 partition 信息。

2. 文档和部分注释仍容易把 CTDG 的 sampler 内部索引误写成 CTDG 外层输入。
   更准确的定位是：DTDG 派生 snapshot/window CSC/CSR 完整子图视图；CTDG 派生 event stream 和 temporal sampling index，输出 MFG/native block。

3. `ExecutionUnit` 新路径尚未接入 `SchedulerSession.run_epoch()`。
   当前正式 train/predict 仍走 `BatchData` 兼容路径，MemShare event path 需要手动调用或后续接入训练 step。

4. `ChunkAssignment.chunk_to_nodes` 和 route artifact 仍是 Python object/list 结构。
   大规模场景下应改成 packed tensor bundle，减少序列化和加载开销。

5. `PropagationPlan` 尚未由模型层 route 填充。
   当前还没有真正覆盖多层 GNN 正向/反向传播中的跨分区消息通信。

6. `MemShareEventEngine.sample()` 目前返回的 `edge_ids` 仍为空。
   后续模型或 memory 更新如果依赖 sampled edge ids，需要从 native block 中补齐。
