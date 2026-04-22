# StarryUniGraph 库文档

## 1. 项目概览

`StarryUniGraph` 是一个面向动态图分布式训练与推理的统一调度库。它用一套统一接口同时支持两类动态图：

- `CTDG`：连续时间动态图，对齐 BTS-MTGNN 风格的采样与 memory 驱动执行
- `DTDG`：离散时间动态图，对齐 Flare 风格的快照与窗口驱动执行

用户层只需要面对一个统一入口，系统内部会根据模型家族自动调度到 CTDG 或 DTDG 对应的运行时后端。

当前实现状态：

- 已完成统一的 `session / provider / config / checkpoint` 架构
- 已完成统一的 `core/protocol.py`
- 已完成 CTDG 与 DTDG 两类 kernel 的统一封装
- 已将 BTS 的 C++ sampler 源码和二进制并入当前仓库，并提供 Python 加载层
- CTDG 在线运行时（`runtime/online/`）已实现：memory bank、历史缓存、采样器、分布式同步
- Flare 高性能组件已全部合并：`PartitionData`、`STGraphLoader`、`RNNStateManager`、route 感知的 pinned-memory 加载、基于 CUDA stream 的异步数据搬运
- DTDG `flare_native` 管线已端到端可运行，支持 DDP 训练

## 2. 设计目标

这个库围绕四个目标设计：

1. 对用户暴露唯一的顶层 API
2. 将"调度层"和"执行层"解耦
3. 保留 CTDG 与 DTDG 原本不同的流水线语义，而不是强行揉成一个伪统一算法
4. 支持逐步把 BTS-MTGNN 和 FlareDTDG 的高性能核心迁移到当前仓库

## 3. 目录结构

### 3.1 顶层入口

- `starry_unigraph/__init__.py` — 重导出 `SchedulerSession`
- `starry_unigraph/session.py` — 统一会话生命周期
- `starry_unigraph/types.py` — 共享数据类型（`DistributedContext`、`SessionContext`、`RuntimeBundle`、`PreparedArtifacts`、`PredictionResult`）
- `starry_unigraph/distributed.py` — 分布式初始化/销毁工具

### 3.2 配置系统

- `starry_unigraph/config/default.yaml` — 默认配置
- `starry_unigraph/config/schema.py` — 配置加载、合并、校验、图模式检测

### 3.3 注册表

- `starry_unigraph/registry/model_registry.py` — 将模型名/家族映射到 `ModelSpec` 与 `graph_mode`
- `starry_unigraph/registry/provider_registry.py` — 将 `graph_mode` 映射到 provider 类
- `starry_unigraph/registry/task_registry.py` — 将任务类型映射到任务适配器

### 3.4 核心执行层

- `starry_unigraph/core/protocol.py` — 统一 kernel 协议（`KernelBatch`、`KernelExecutor`、`PipelineTrace` 等）
- `starry_unigraph/core/ctdg_kernel.py` — CTDG 专属 kernel 数据类型与执行器
- `starry_unigraph/core/dtdg_kernel.py` — DTDG 专属 kernel 数据类型与执行器
- `starry_unigraph/core/data_split.py` — train/val/test 切分比例归一化与区间计算

### 3.5 数据模块

- `starry_unigraph/data/raw_temporal.py` — 原始事件加载（`RawTemporalEvents`、CSV/边文件/mock 加载器）
- `starry_unigraph/data/partition.py` — 分区数据结构（`TensorData`、`RouteData`、`PartitionData`）

### 3.6 Provider 层

- `starry_unigraph/providers/common.py` — 共享 artifact 工具、`BaseProvider` 抽象类
- `starry_unigraph/providers/ctdg.py` — `CTDGPreprocessor`、`CTDGProvider`
- `starry_unigraph/providers/dtdg.py` — `DTDGProvider`（调度 Flare 管线）
- `starry_unigraph/providers/dtdg_loaders.py` — `FlareRuntimeLoader`（封装 `STGraphLoader` 与 split 管理）
- `starry_unigraph/providers/dtdg_preprocess.py` — `BaseDTDGPreprocessor`、`FlareDTDGPreprocessor`
- `starry_unigraph/providers/dtdg_train.py` — Flare 训练辅助函数（`init_flare_training`、`run_flare_train_step` 等）

### 3.7 预处理

- `starry_unigraph/preprocess/base.py` — 抽象 `GraphPreprocessor`、`ArtifactLayout`、`ArtifactOutput`
- `starry_unigraph/preprocess/dtdg_prepare.py` — DTDG 数据准备（快照提取、METIS 分区、`PartitionData` 生成）

### 3.8 运行时

- `starry_unigraph/runtime/base.py` — 抽象协议（`RuntimeProtocol`、`RuntimeAdapter`、`ExecutionAdapter`、`GraphProvider`）

#### Flare 运行时（`runtime/flare/`）

- `starry_unigraph/runtime/flare/loader.py` — `STGraphLoader`（快照迭代 + CUDA stream 预取）
- `starry_unigraph/runtime/flare/state.py` — `RNNStateManager`、`STGraphBlob`、`STWindowState`
- `starry_unigraph/runtime/flare/route.py` — `Route`、`RouteAgent`、可微分 all-to-all 交换（含 autograd 钩子）
- `starry_unigraph/runtime/flare/models.py` — Flare DTDG 模型（`FlareEvolveGCN`、`FlareTGCN`、`FlareMPNNLSTM`、`GCNStack`）
- `starry_unigraph/runtime/flare/training.py` — Flare 训练/评估/预测步函数（含 DDP 支持）

#### 在线运行时（`runtime/online/`）

- `starry_unigraph/runtime/online/data.py` — `CTDGDataBatch`、`TGTemporalDataset`
- `starry_unigraph/runtime/online/memory.py` — `CTDGMemoryBank`（节点级 memory + K-slot mailbox + 分布式异步同步）
- `starry_unigraph/runtime/online/cache.py` — `CTDGHistoricalCache`、`AdaParameter`（余弦距离变化检测）
- `starry_unigraph/runtime/online/sampler.py` — `NativeTemporalSampler`、`CTDGSampleOutput`
- `starry_unigraph/runtime/online/models.py` — `CTDGMemoryUpdater`、`CTDGLinkPredictor`、`CTDGModelOutput`
- `starry_unigraph/runtime/online/route.py` — `CTDGFeatureRoute`、`AsyncExchangeHandle`
- `starry_unigraph/runtime/online/runtime.py` — `CTDGOnlineRuntime`（完整 CTDG 训练/评估/预测编排）

### 3.9 后端（兼容层）

- `starry_unigraph/backends/__init__.py` — 重导出常用类
- `starry_unigraph/backends/flare/__init__.py` — 从 `runtime/flare/` 重导出
- `starry_unigraph/backends/ctdg/__init__.py` — 从 `runtime/online/` 重导出

### 3.10 任务

- `starry_unigraph/tasks/base.py` — `BaseTaskAdapter`
- `starry_unigraph/tasks/temporal_link_prediction.py` — `TemporalLinkPredictionTaskAdapter`
- `starry_unigraph/tasks/node_regression.py` — `NodeRegressionTaskAdapter`

### 3.11 原生扩展与 vendor 代码

- `starry_unigraph/native/bts_sampler.py` — BTS 采样器 Python 封装（`BTSNativeSampler`、`build_temporal_neighbor_block`）
- `starry_unigraph/lib/loader.py` — `load_bts_sampler_module()`（延迟加载 `.so`）
- `starry_unigraph/lib/libstarrygl_sampler.so` — 预编译 BTS 采样器二进制
- `starry_unigraph/vendor/bts_sampler/` — BTS C++ 源码与 CMake 构建

### 3.12 Checkpoint 与 CLI

- `starry_unigraph/checkpoint/io.py` — `save_checkpoint()`、`load_checkpoint()`
- `starry_unigraph/cli/main.py` — 命令行入口（prepare/train/predict/resume）
- `tests/test_session.py` — session 级测试

## 4. 总体架构

当前库分为三个主层加两个运行时后端。

### 4.1 Session 层

用户主要通过 `session.py` 中的 `SchedulerSession` 交互。

主要入口：

- `SchedulerSession.from_config(...)` — 加载配置、推断图模式、构造 provider
- `prepare_data()` — 触发预处理、写出 artifacts
- `build_runtime()` — 初始化模型、优化器、加载器、runtime state
- `run_epoch(split)` — 按 split 迭代一个 epoch
- `run_task()` — 完整 train/eval 主循环
- `evaluate(split)` — 单次评估
- `predict(split)` — 推理并返回 `PredictionResult`
- `save_checkpoint(path)` / `load_checkpoint(path)` — 持久化/恢复

这一层负责：

- 读取并校验配置
- 推断 `graph_mode`
- 解析 provider
- 组织统一的 train/eval/predict/checkpoint 生命周期

不负责具体采样、快照加载、memory 更新等执行细节。

### 4.2 Provider 层

Provider 负责把配置映射到运行时后端。

- `providers/ctdg.py` — 通过 `CTDGOnlineRuntime` 编排
- `providers/dtdg.py` — 通过 `FlareRuntimeLoader` + Flare 训练步函数编排

职责：

- 生成预处理产物目录与 metadata
- 构造 family-specific 的 partition/route/runtime 对象
- 对 session 暴露 iterator
- 将 step 执行委托给运行时后端

不负责：

- 采样实现（属于 `runtime/online/`）
- 快照执行（属于 `runtime/flare/`）
- memory 更新逻辑（属于 `runtime/online/`）

### 4.3 Kernel 层

Kernel 层定义执行协议与 family-specific 的数据类型。

- `core/protocol.py` 定义统一执行协议
- `core/ctdg_kernel.py` 定义 CTDG batch/state/result 类型
- `core/dtdg_kernel.py` 定义 DTDG batch/state/result 类型

### 4.4 运行时后端

真正的执行逻辑位于两个运行时后端：

- `runtime/flare/` — Flare 风格 DTDG 执行（快照加载、GCN/RNN 模型、route 交换、DDP 训练）
- `runtime/online/` — BTS 风格 CTDG 执行（时序采样、memory bank、历史缓存、分布式同步）

`backends/` 包提供向后兼容的重导出。

## 5. 统一协议层

统一协议定义在 `core/protocol.py`。

核心抽象：

- `KernelBatch` — batch 基类（`index`、`split`、`chain`、`to_payload()`）
- `KernelRuntimeState` — 多步执行中持续演化的状态
- `KernelResult` — 步执行结果（`predictions`、`targets`、`loss`、`meta`）
- `KernelExecutor` — 统一执行器接口（`iter_batches`、`execute_train/eval/predict`、`dump_state`）
- `PipelineTrace` — 记录阶段级执行轨迹，支持异步流程管理
- `AsyncStageHandle` — 异步阶段跟踪（`token`、`name`、`status`、`depends_on`）
- `StateHandle` — 状态容器标识（CTDG: 节点 memory；DTDG: 窗口状态）
- `StateDelta` — 描述当前步的状态变化
- `StateWriteback` — 组合 `StateHandle` + `StateDelta` + 版本号

流水线阶段记录：

- CTDG：`sample`、`feature_fetch`、`state_fetch`、`memory_updater`、`neighbor_attention_aggregate`、`message_generate`、`state_transition`、`state_writeback`
- DTDG：`load_snapshot`、`route_apply`、`state_fetch`、`state_transition`、`state_writeback`

## 6. CTDG 模块说明

### 6.1 核心 kernel 类型（`core/ctdg_kernel.py`）

- `FeatureRoutePlan` — route_type, fanout, feature_keys
- `StateSyncPlan` — 同步模式与版本计数
- `CTDGPartitionBook` — 分区元数据
- `CTDGMailboxState` — memory/mailbox 版本追踪
- `CTDGBatch(KernelBatch)` — 含采样计划的 batch
- `CTDGPreparedBatch` — 物化后的 batch
- `CTDGStepResult(KernelResult)` — 步输出
- `CTDGRuntimeState(KernelRuntimeState)` — 运行时游标

### 6.2 在线运行时（`runtime/online/`）

CTDG 执行路径在 `runtime/online/` 中实现：

**数据管线：**

- `TGTemporalDataset` — 加载原始事件，提供分布式感知的 batch 迭代（含 train/val/test 切分）
- `CTDGDataBatch` — 事件小批量（src, dst, ts, edge_feat）

**采样：**

- `NativeTemporalSampler` — 封装 BTS C++ 采样器进行多跳时序邻居采样
- `CTDGSampleOutput` — 采样得到的邻居、边、时间戳

**Memory 与状态：**

- `CTDGMemoryBank` — 节点级隐藏 memory + K-slot mailbox + 通过 `all_to_all_single` 进行分布式异步同步
- `CTDGHistoricalCache` — 缓存上次同步的 memory，使用余弦距离过滤（`AdaParameter`）跳过冗余同步

**模型：**

- `CTDGMemoryUpdater` — 基于 GRU 的 mailbox 历史更新
- `CTDGLinkPredictor` — 时序 transformer 注意力机制的链接预测

**路由：**

- `CTDGFeatureRoute` — 跨分区特征交换
- `AsyncExchangeHandle` — 异步 send/recv 句柄

**编排：**

- `CTDGOnlineRuntime` — 完整 CTDG 训练/评估/预测循环（memory 更新 → 采样 → 卷积 → 评分 → 分布式同步）

### 6.3 CTDG 执行流水线

CTDG 运行时保留 BTS 风格的阶段顺序：

1. `sample` — 通过 BTS 原生采样器进行时序邻居采样
2. `feature_fetch` — route 感知的特征获取
3. `state_fetch` — 收集 memory/mailbox 状态
4. `memory_updater` — GRU 更新 mailbox
5. `neighbor_attention_aggregate` — 对采样的时序邻居做注意力聚合
6. `message_generate` — 生成状态更新消息
7. `state_transition` — 更新节点 memory
8. `state_writeback` — 写回分布式 memory bank

### 6.4 CTDG 预处理产物

- `artifacts/{dataset}/meta/artifacts.json`
- `artifacts/{dataset}/partitions/manifest.json`
- `artifacts/{dataset}/routes/manifest.json`
- `artifacts/{dataset}/sampling/index.json`

### 6.5 BTS 原生采样器

- 源码：`vendor/bts_sampler/`（C++ + CMake）
- 预编译二进制：`lib/libstarrygl_sampler.so`
- 加载入口：`lib/loader.py`（`load_bts_sampler_module()`）
- Python 封装：`native/bts_sampler.py`（`BTSNativeSampler`、`build_temporal_neighbor_block`、`is_bts_sampler_available`）

通过 `CTDGSamplerCore.attach_native_sampler(...)` 接入，由 `NativeTemporalSampler` 在在线运行时中消费。

## 7. DTDG 模块说明

### 7.1 核心 kernel 类型（`core/dtdg_kernel.py`）

- `SnapshotRoutePlan` — route_type, cache_policy
- `DTDGPartitionBook` — 含 snapshot_count 的分区元数据
- `DTDGWindowState` — 窗口追踪
- `DTDGBatch(KernelBatch)` — 含 adjacency、features、graph、graph_meta 的 batch
- `DTDGStepResult(KernelResult)` — 步输出
- `DTDGRuntimeState(KernelRuntimeState)` — snapshot 游标

### 7.2 数据结构（`data/partition.py`）

- `TensorData` — 压缩的变长张量存储（CSR 风格 ptr/ind）
- `RouteData` — 每快照路由描述（send/recv sizes, send_index）
- `PartitionData` — 完整分区容器（node_data, edge_data, labels, routes, chunks）。支持 `save()`/`load()`、`pin_memory()`、`to(device)`、`to_blocks()` 转 DGL block。

### 7.3 Flare 运行时（`runtime/flare/`）

DTDG 执行路径在 `runtime/flare/` 中完整实现：

**数据加载：**

- `STGraphLoader` — 遍历 `PartitionData` 快照并转为 DGL block。使用 `pin_memory()` + 独立 `torch.cuda.Stream` 进行异步 CPU→GPU 数据搬运。工厂方法：`STGraphLoader.from_partition_data(data, device)`。

**状态管理：**

- `RNNStateManager` — 管理滑动窗口内的图序列与每快照 RNN 状态。为每个图 patch `flare_fetch_state()` / `flare_store_state()` 方法。支持 "pad" 模式（零填充新状态）和 "mix" 模式（与前序混合）。
- `STGraphBlob` — 封装图序列 + `RNNStateManager`。可迭代。属性 `current_graph` 返回窗口中最后一个图。
- `STWindowState` — 窗口大小与快照追踪元数据。

**路由：**

- `Route` — 每快照可微分 all-to-all 特征交换。通过 `RouteSendFunction` / `RouteRecvFunction` 兼容 autograd。
- `RouteAgent` — 执行实际的 `dist.all_to_all_single` 通信。

**模型：**

- `GCNStack` — 多层 GCN 消息传递
- `FlareEvolveGCN` — EvolveGCN-H（GRU 演化 GCN 权重）
- `FlareTGCN` — Temporal GCN（GCN + GRU）
- `FlareMPNNLSTM` — MPNN-LSTM（GCN + 两个自定义 `_LSTMCell`）。状态为 4-tuple `(h1,c1,h2,c2)`，通过 Python tuple 拼接实现。
- `build_flare_model(name, ...)` — 工厂函数

**训练：**

- `init_flare_training(config, partition_data)` — 构建模型、优化器、DDP 包装、计算分区 loss scale
- `run_flare_train_step(blob, model, optimizer, ...)` — forward + backward + optimizer step
- `run_flare_eval_step(blob, model, ...)` — forward（无梯度）
- `run_flare_predict_step(blob, model, ...)` — forward（无梯度），返回预测结果

### 7.4 Provider 集成（`providers/dtdg*.py`）

- `FlareDTDGPreprocessor` — 通过 METIS 分区与快照提取生成 `PartitionData` 文件（`part_NNN.pth`）
- `FlareRuntimeLoader` — 封装 `STGraphLoader` 与 train/eval/predict split 管理。训练时 yield `STGraphBlob`；评估/预测时 yield `DTDGBatch`（单帧）。
- `DTDGProvider` — 端到端编排 Flare 管线：预处理 → runtime 构建 → 迭代器 → step 执行

### 7.5 DTDG 执行流水线

Flare-native 管线保留以下阶段顺序：

1. `load_snapshot` — `STGraphLoader` 通过 CUDA stream 异步获取并传输快照到 GPU
2. `route_apply` — `Route.forward()` 执行可微分 all-to-all 特征交换
3. `state_fetch` — `RNNStateManager` 通过 `flare_fetch_state()` 提供每快照 RNN 状态
4. `state_transition` — 模型前向（GCN + RNN 层）
5. `state_writeback` — `flare_store_state()` 将更新后的 RNN 状态写回 manager

### 7.6 DTDG 预处理产物

- `artifacts/{dataset}/meta/artifacts.json`
- `artifacts/{dataset}/partitions/manifest.json`
- `artifacts/{dataset}/routes/manifest.json`
- `artifacts/{dataset}/flare/manifest.json`、`flare/part_NNN.pth`（Flare-native）
- `artifacts/{dataset}/snapshots/manifest.json`（chunked，如适用）

### 7.7 DTDG 数据准备（`preprocess/dtdg_prepare.py`）

完整 DTDG 数据准备管线：

- `build_partition_input_from_raw_dataset()` — 从原始事件中提取快照与元数据
- `build_dtdg_partitions()` — 执行 METIS 或随机图分区
- `build_flare_partition_data_list()` — 创建每分区的 `PartitionData` 对象（含本地边索引、节点特征、标签与路由描述）

### 7.8 数据切分工具（`core/data_split.py`）

- `normalize_split_ratio(split_ratio)` — 确保 train+val+test 之和为 1.0
- `split_bounds(total, split, split_ratio)` — 返回指定 split 的 `(start, end)` 索引区间

## 8. 配置系统

配置系统定义在 `config/schema.py`，默认值在 `config/default.yaml`。

### 8.1 主要配置段

- `model` — name, family, task, hidden_dim, memory (CTDG), window (DTDG)
- `data` — root, name, format, graph_mode, split_ratio
- `train` — epochs, batch_size, snaps, eval_interval, lr
- `runtime` — backend, device, cache, checkpoint
- `dtdg` — pipeline, chunk_order, chunk_decay, num_full_snaps
- `ctdg` — pipeline, mailbox_slots, historical_alpha, async_sync, ada_param_enabled, dropout
- `preprocess` — cluster 与 chunk 设置
- `sampling` — neighbor_limit, strategy, history, neg_sampling
- `graph` — storage, partition, route
- `dist` — backend, world_size, rank, local_rank, master_addr, master_port

### 8.2 graph mode 选择

`model.family` 会通过 `ModelRegistry` 解析：

- CTDG 家族：`tgn`、`dyrep`、`jodie`、`tgat`、`apan`
- DTDG 家族：`evolvegcn`、`tgcn`、`mpnn_lstm`、`gcn`

然后驱动 provider 与 kernel 的选择。

### 8.3 校验行为

schema 层：

- 从 `default.yaml` 合并默认配置
- 校验必填路径（`model.name`、`data.root` 等）
- 自动推断 graph mode
- 对非激活字段发 warning（如 DTDG 模式下的 CTDG 配置项）
- 对关键缺失项抛出异常

## 9. 注册表系统

注册表的作用是把"名字"和"实现"解耦。

### 9.1 ModelRegistry

将 `model.name` 或 `model.family` 映射到 `ModelSpec`（含 name, family, graph_mode）。

### 9.2 ProviderRegistry

将 `ctdg` 映射为 `CTDGProvider`，`dtdg` 映射为 `DTDGProvider`。

### 9.3 TaskRegistry

将任务类型映射到任务适配器类：

- `"temporal_link_prediction"` → `TemporalLinkPredictionTaskAdapter`
- `"snapshot_node_regression"` → `NodeRegressionTaskAdapter`

## 10. Runtime 与 Checkpoint

运行时抽象定义在 `runtime/base.py`，提供以下抽象协议：

- `RuntimeProtocol` — `iter_batches()`、`step()`、`state_dict()`、`load_state_dict()`
- `RuntimeAdapter` — 模型/优化器初始化、runtime 状态 load/dump
- `ExecutionAdapter` — `train_step()`、`eval_step()`、`predict_step()`
- `GraphProvider` — 组合 adapter + execution，带 `graph_mode` 属性

checkpoint 读写位于 `checkpoint/io.py`。

### 10.1 当前 checkpoint 内容

- `model_state`
- `optimizer_state`
- `scheduler_state`
- `runtime_state`
- `provider_meta`
- `config`
- `epoch`
- `global_step`

### 10.2 当前 runtime state

CTDG：`memory_state`、`mailbox_state`、`sampler_state`、`executor_state`

DTDG：`window_state`、`snapshot_state`、`route_cache`、`executor_state`

## 11. 预处理层

预处理抽象定义在 `preprocess/base.py`：

- `ArtifactLayout` — 根目录 + 子目录映射
- `ArtifactOutput` — relative_path, payload, serializer (json/torch)
- `ArtifactPayload` — provider_meta + outputs 列表
- `GraphPreprocessor`（抽象）— `prepare_raw()`、`build_partitions()`、`build_runtime_artifacts()`、`run()`

具体预处理器：

- `CTDGPreprocessor` — 将原始时序事件准备为分区 manifest + feature route plan
- `BaseDTDGPreprocessor` / `FlareDTDGPreprocessor` — 完整 DTDG 预处理管线：原始事件加载 → 快照提取 → METIS 分区 → `PartitionData` 生成（`part_NNN.pth` 文件）

## 12. 分布式支持

分布式工具位于 `distributed.py`：

- `apply_distributed_env()` — 从环境变量读取 `WORLD_SIZE`/`RANK`/`LOCAL_RANK`（由 `torchrun` 设置），合并到配置
- `build_distributed_context()` — 从配置创建 `DistributedContext`
- `initialize_distributed()` — 调用 `dist.init_process_group()`，设置 CUDA device
- `finalize_distributed()` — 调用 `dist.destroy_process_group()`

DDP 训练：

- Flare：`init_flare_training()` 将模型包装为 `DistributedDataParallel`，通过 `all_reduce` 计算分区 loss scale
- Online：`CTDGOnlineRuntime` 通过 `all_to_all_single` 处理分布式 memory 同步

## 13. CLI

命令行入口位于 `cli/main.py`。

支持命令：

- `prepare` — 仅预处理
- `train` — 预处理 + 构建 + 训练
- `predict` — 推理
- `resume` — 从 checkpoint 恢复并继续

示例：

```bash
python -m starry_unigraph.cli prepare --config path/to/config.yaml
python -m starry_unigraph.cli train --config path/to/config.yaml
python -m starry_unigraph.cli predict --config path/to/config.yaml --split test
python -m starry_unigraph.cli resume --config path/to/config.yaml --checkpoint path/to/ckpt.pkl
```

## 14. 当前运行流程

### 14.1 创建 session

```python
from starry_unigraph import SchedulerSession

session = SchedulerSession.from_config("starry_unigraph/config/default.yaml")
```

### 14.2 预处理与构建

```python
session.prepare_data()
session.build_runtime()
```

### 14.3 训练

```python
summary = session.run_epoch(split="train")
```

### 14.4 完整训练循环

```python
result = session.run_task()
```

### 14.5 预测

```python
result = session.predict(split="test")
```

### 14.6 保存与恢复

```python
session.save_checkpoint("checkpoint.pkl")
session.load_checkpoint("checkpoint.pkl")
```

## 15. 流水线语义

当前库的核心原则之一，是统一接口但不抹平原始执行语义。

### 15.1 CTDG 的语义

保留：

- 基于 BTS 原生采样器的采样优先执行
- route 感知的跨分区特征获取
- 带分布式异步同步的 memory/mailbox 更新
- 基于自适应余弦距离过滤的历史缓存

### 15.2 DTDG 的语义

保留：

- 基于 CUDA stream 预取的快照加载
- 可微分 route 的 all-to-all 特征交换
- 滑动窗口 RNN 状态管理（pad/mix 模式）
- 分区级 loss 缩放的平衡 DDP 训练

## 16. 当前已从 BTS 和 Flare 合并的内容

### 已合并

- 统一 session/provider/config/checkpoint 架构
- BTS sampler 源码与二进制资产
- BTS sampler Python 封装与加载入口
- 统一 kernel 协议
- BTS 风格 CTDG 内核与在线运行时（memory bank、历史缓存、采样器、分布式同步）
- Flare 风格 DTDG 内核
- Flare `PartitionData`（CSR 风格张量存储）
- Flare `STGraphLoader`（CUDA stream 预取）
- Flare `RNNStateManager` 与 `STGraphBlob`
- Flare 可微分 `Route`（含 autograd 钩子）
- Flare DTDG 模型（`FlareEvolveGCN`、`FlareTGCN`、`FlareMPNNLSTM`）
- Flare DDP 训练（含分区 loss 缩放）
- DTDG 预处理（METIS 分区 + `PartitionData` 生成）

### 尚未完全合并

- BTS 原生采样器驱动的完整端到端 CTDG 训练路径（采样器已集成但 kernel 自动调度为部分完成）
- BTS memory/mailbox 与上游的完整实现对齐
- DTDG `chunked` 管线的高性能后端（自适应 chunk 准备已存在但尚未完全集成）

## 17. 测试覆盖

测试位于 `tests/test_session.py`。

当前覆盖：

- provider 解析
- artifact 生成
- runtime 构建
- CTDG / DTDG pipeline 执行
- checkpoint 保存恢复
- predict 输出形状
- inactive field warning
- 缺失配置路径报错
- artifact version mismatch 提前失败
- 缺失 route manifest 提前失败
- BTS native sampler 模块可用性

## 18. 总结

当前 `StarryUniGraph` 已经具备：

- 一个统一的用户入口
- 一套统一的配置系统
- 一套统一的 checkpoint 格式
- 一套统一的执行协议
- 两套 family-specific 运行时后端（Flare 用于 DTDG，Online 用于 CTDG）
- 完整的 Flare 高性能数据路径（PartitionData、STGraphLoader、RNNStateManager、可微分路由、DDP 训练）
- CTDG 在线运行时（memory bank、采样、分布式同步）

该库在保留 CTDG 与 DTDG 原始流水线语义的同时，提供了统一的结构化工程入口。
