# StarryUniGraph 库文档

## 1. 项目概览

`StarryUniGraph` 是一个面向动态图分布式训练与推理的统一调度库。它希望用一套统一接口同时支持两类动态图：

- `CTDG`：连续时间动态图，当前对齐 BTS-MTGNN 风格的采样与 memory 驱动执行
- `DTDG`：离散时间动态图，当前对齐 Flare 风格的快照与窗口驱动执行

用户层只需要面对一个统一入口，系统内部会根据模型家族自动调度到 CTDG 或 DTDG 对应的执行内核。

当前实现状态如下：

- 已完成统一的 `session / provider / config / checkpoint` 架构
- 已完成统一的 `core/protocol.py`
- 已完成 CTDG 与 DTDG 两类 kernel 的统一封装
- 已将 BTS 的 C++ sampler 源码和二进制并入当前仓库，并提供 Python 加载层
- 尚未完成 Flare `PartitionData / STGraphLoader` 高性能数据路径的迁移

## 2. 设计目标

这个库目前围绕四个目标设计：

1. 对用户暴露唯一的顶层 API
2. 将“调度层”和“执行层”解耦
3. 保留 CTDG 与 DTDG 原本不同的流水线语义，而不是强行揉成一个伪统一算法
4. 支持逐步把 BTS-MTGNN 和 FlareDTDG 的高性能核心迁移到当前仓库

## 3. 目录结构

### 3.1 顶层入口

- [starry_unigraph/__init__.py](/home/zlj/StarryUniGraph/starry_unigraph/__init__.py)
- [starry_unigraph/session.py](/home/zlj/StarryUniGraph/starry_unigraph/session.py)
- [starry_unigraph/types.py](/home/zlj/StarryUniGraph/starry_unigraph/types.py)

### 3.2 配置系统

- [starry_unigraph/config/default.yaml](/home/zlj/StarryUniGraph/starry_unigraph/config/default.yaml)
- [starry_unigraph/config/schema.py](/home/zlj/StarryUniGraph/starry_unigraph/config/schema.py)

### 3.3 注册表

- [starry_unigraph/registry/model_registry.py](/home/zlj/StarryUniGraph/starry_unigraph/registry/model_registry.py)
- [starry_unigraph/registry/provider_registry.py](/home/zlj/StarryUniGraph/starry_unigraph/registry/provider_registry.py)
- [starry_unigraph/registry/task_registry.py](/home/zlj/StarryUniGraph/starry_unigraph/registry/task_registry.py)

### 3.4 核心执行层

- [starry_unigraph/core/protocol.py](/home/zlj/StarryUniGraph/starry_unigraph/core/protocol.py)
- [starry_unigraph/core/ctdg_kernel.py](/home/zlj/StarryUniGraph/starry_unigraph/core/ctdg_kernel.py)
- [starry_unigraph/core/dtdg_kernel.py](/home/zlj/StarryUniGraph/starry_unigraph/core/dtdg_kernel.py)
- [starry_unigraph/core/ctdg.py](/home/zlj/StarryUniGraph/starry_unigraph/core/ctdg.py)
- [starry_unigraph/core/dtdg.py](/home/zlj/StarryUniGraph/starry_unigraph/core/dtdg.py)

### 3.5 Provider 层

- [starry_unigraph/providers/common.py](/home/zlj/StarryUniGraph/starry_unigraph/providers/common.py)
- [starry_unigraph/providers/ctdg.py](/home/zlj/StarryUniGraph/starry_unigraph/providers/ctdg.py)
- [starry_unigraph/providers/dtdg.py](/home/zlj/StarryUniGraph/starry_unigraph/providers/dtdg.py)

### 3.6 原生扩展与 vendor 代码

- [starry_unigraph/native/bts_sampler.py](/home/zlj/StarryUniGraph/starry_unigraph/native/bts_sampler.py)
- [starry_unigraph/lib/loader.py](/home/zlj/StarryUniGraph/starry_unigraph/lib/loader.py)
- [starry_unigraph/lib/libstarrygl_sampler.so](/home/zlj/StarryUniGraph/starry_unigraph/lib/libstarrygl_sampler.so)
- [starry_unigraph/vendor/bts_sampler/CMakeLists.txt](/home/zlj/StarryUniGraph/starry_unigraph/vendor/bts_sampler/CMakeLists.txt)

### 3.7 任务、运行时与持久化

- [starry_unigraph/tasks/base.py](/home/zlj/StarryUniGraph/starry_unigraph/tasks/base.py)
- [starry_unigraph/tasks/temporal_link_prediction.py](/home/zlj/StarryUniGraph/starry_unigraph/tasks/temporal_link_prediction.py)
- [starry_unigraph/tasks/node_regression.py](/home/zlj/StarryUniGraph/starry_unigraph/tasks/node_regression.py)
- [starry_unigraph/runtime/base.py](/home/zlj/StarryUniGraph/starry_unigraph/runtime/base.py)
- [starry_unigraph/checkpoint/io.py](/home/zlj/StarryUniGraph/starry_unigraph/checkpoint/io.py)

### 3.8 CLI 与测试

- [starry_unigraph/cli/main.py](/home/zlj/StarryUniGraph/starry_unigraph/cli/main.py)
- [tests/test_session.py](/home/zlj/StarryUniGraph/tests/test_session.py)

## 4. 总体架构

当前库可以理解成三层。

### 4.1 Session 层

用户主要通过 [session.py](/home/zlj/StarryUniGraph/starry_unigraph/session.py) 中的 `SchedulerSession` 交互。

主要入口：

- `SchedulerSession.from_config(...)`
- `prepare_data()`
- `build_runtime()`
- `train_step()`
- `run_epoch()`
- `run_task()`
- `predict()`
- `save_checkpoint()`
- `load_checkpoint()`

这一层负责：

- 读取配置
- 校验配置
- 推断 `graph_mode`
- 解析 provider
- 组织统一的 train/eval/predict/checkpoint 生命周期

这一层不负责：

- 具体采样
- 具体快照加载
- memory 更新
- 稀疏算子执行

### 4.2 Provider 层

Provider 负责把配置映射成具体 kernel。

- [providers/ctdg.py](/home/zlj/StarryUniGraph/starry_unigraph/providers/ctdg.py)
- [providers/dtdg.py](/home/zlj/StarryUniGraph/starry_unigraph/providers/dtdg.py)

它们的职责是：

- 生成预处理产物目录与 metadata
- 构造 family-specific 的 partition/route/runtime 对象
- 根据 config 创建对应 kernel
- 对 session 暴露 iterator
- 将 kernel state 映射成统一 runtime state 和 checkpoint state

它们现在已经尽量变薄，不再持有核心执行逻辑。

### 4.3 Kernel 层

Kernel 层是真正的执行层。

- [protocol.py](/home/zlj/StarryUniGraph/starry_unigraph/core/protocol.py)
- [ctdg_kernel.py](/home/zlj/StarryUniGraph/starry_unigraph/core/ctdg_kernel.py)
- [dtdg_kernel.py](/home/zlj/StarryUniGraph/starry_unigraph/core/dtdg_kernel.py)

真正的流水线阶段都在这里定义与执行。

## 5. 统一协议层

统一协议定义在 [protocol.py](/home/zlj/StarryUniGraph/starry_unigraph/core/protocol.py)。

核心抽象如下：

- `KernelBatch`
- `KernelRuntimeState`
- `KernelResult`
- `KernelExecutor`
- `PipelineTrace`
- `AsyncStageHandle`
- `StateHandle`
- `StateDelta`
- `StateWriteback`

### 5.1 KernelBatch

表示一个执行单元。

共同约定：

- 必须有 `index`
- 必须有 `split`
- 必须有 `chain`
- 必须能通过 `to_payload()` 序列化

### 5.2 KernelRuntimeState

表示 kernel 在多步执行中持续演化的状态。

例如：

- CTDG 的 sampler cursor
- DTDG 的 snapshot cursor
- 上一轮 prediction
- 当前运行 split

### 5.3 KernelResult

表示一步执行结果。

统一结果形状通常包含：

- `predictions`
- `targets`
- `loss`
- `meta`

### 5.4 KernelExecutor

这是统一执行器接口，定义了：

- `iter_batches(split, count)`
- `execute_train(batch)`
- `execute_eval(batch)`
- `execute_predict(batch)`
- `dump_state()`

这就是当前内部统一的关键位置。

### 5.5 PipelineTrace

`PipelineTrace` 用来显式记录每个阶段的执行轨迹，保留原始 pipeline 语义。

它现在不只是“记录日志”，还承担异步流程管理语义：

- 记录同步阶段顺序
- 记录异步阶段 token
- 记录异步阶段状态：`pending` / `completed` / `failed`
- 记录异步依赖关系

它适合用来管理：

- 异步 feature fetch
- 异步 state writeback
- 后续可能接入的异步通信、异步 route、异步参数更新

当前：

- CTDG 会记录 `sample`、`feature_fetch`、`state_fetch`、`memory_updater`、`neighbor_attention_aggregate`、`message_generate`、`state_transition`、`state_writeback`
- DTDG 会记录 `load_snapshot`、`route_apply`、`state_fetch`、`state_transition`、`state_writeback`

### 5.6 AsyncStageHandle

`AsyncStageHandle` 用于描述一个可异步管理的阶段。

当前包含：

- `token`
- `name`
- `status`
- `payload`
- `depends_on`

它的作用是让 `PipelineTrace` 不仅能记录“做过什么”，还能表达：

- 哪些阶段是异步候选
- 哪些阶段依赖前序阶段
- 哪些异步任务已经完成或失败
### 5.7 StateHandle

`StateHandle` 用于描述“状态存放在哪里、属于谁、作用范围是什么”。

当前典型用法：

- CTDG：表示节点级 memory/mailbox 状态容器
- DTDG：表示窗口级 temporal state 容器

### 5.8 StateDelta

`StateDelta` 用于描述“当前 step 对状态造成了什么变化”。

当前典型用法：

- CTDG：描述 memory updater、message passing、message generate 产生的状态变化
- DTDG：描述 snapshot propagation 与 temporal fusion 产生的状态变化

### 5.9 StateWriteback

`StateWriteback` 用于描述“如何把状态变化写回运行时”。

统一语义是：

- 先定位状态容器 `StateHandle`
- 再描述变化内容 `StateDelta`
- 最后记录写回版本或轮次

## 6. CTDG 模块说明

CTDG 执行核心位于 [ctdg_kernel.py](/home/zlj/StarryUniGraph/starry_unigraph/core/ctdg_kernel.py)。

### 6.1 主要数据结构

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

### 6.2 CTDG 的执行流水线

当前 CTDG kernel 保留 BTS 风格的阶段顺序：

1. `sample`
2. `feature_fetch`
3. `state_fetch`
4. `memory_updater`
5. `neighbor_attention_aggregate`
6. `message_generate`
7. `state_transition`
8. `state_writeback`

执行主类是 `CTDGKernel`。

主要方法：

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

### 6.3 CTDG 的预处理产物

CTDG provider 会写出以下目录：

- `artifacts/{dataset}/meta/artifacts.json`
- `artifacts/{dataset}/partitions/manifest.json`
- `artifacts/{dataset}/routes/manifest.json`
- `artifacts/{dataset}/sampling/index.json`

这些文件会记录：

- graph mode
- partition algorithm
- number of parts
- feature route plan
- state sync plan
- sampling lookup metadata

### 6.4 BTS 原生采样器接入状态

当前仓库已经把 BTS 的 sampler 合并进来了：

- 源码在 [vendor/bts_sampler](/home/zlj/StarryUniGraph/starry_unigraph/vendor/bts_sampler)
- 预编译二进制在 [libstarrygl_sampler.so](/home/zlj/StarryUniGraph/starry_unigraph/lib/libstarrygl_sampler.so)
- Python 加载入口在 [loader.py](/home/zlj/StarryUniGraph/starry_unigraph/lib/loader.py)
- Python 封装在 [bts_sampler.py](/home/zlj/StarryUniGraph/starry_unigraph/native/bts_sampler.py)

当前已提供：

- `is_bts_sampler_available()`
- `build_temporal_neighbor_block(...)`
- `BTSNativeSampler`

当前尚未完全打通：

- `CTDGKernel` 默认执行路径还没有完全切换到 BTS 原生 sampler 输出
- 还没有把 BTS 的 memory/mailbox 完整关键路径全部迁进仓库

所以当前状态是：

- “原生 sampler 资产和加载层”已经合并
- “端到端高性能 CTDG 内核”还没有完全完成

## 7. DTDG 模块说明

DTDG 执行核心位于 [dtdg_kernel.py](/home/zlj/StarryUniGraph/starry_unigraph/core/dtdg_kernel.py)。

### 7.1 主要数据结构

- `SnapshotRoutePlan`
- `DTDGPartitionBook`
- `DTDGWindowState`
- `DTDGBatch`
- `DTDGStepResult`
- `DTDGRuntimeState`
- `DTDGSnapshotCore`
- `DTDGKernel`

### 7.2 DTDG 的执行流水线

当前 DTDG kernel 保留 Flare 风格的阶段顺序：

1. `load_snapshot`
2. `route_apply`
3. `state_fetch`
4. `state_transition`
5. `state_writeback`

执行主类是 `DTDGKernel`。

主要方法：

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

### 7.3 DTDG 的预处理产物

DTDG provider 会写出：

- `artifacts/{dataset}/meta/artifacts.json`
- `artifacts/{dataset}/partitions/manifest.json`
- `artifacts/{dataset}/routes/manifest.json`
- `artifacts/{dataset}/snapshots/index.json`

这些文件记录：

- graph mode
- partition algorithm
- snapshot count
- route plan
- window metadata

### 7.4 Flare 高性能链路状态

Flare 的高性能预处理和加载链路目前还没有完全迁进当前仓库。

尚未迁移的核心包括：

- `PartitionData`
- `STGraphLoader`
- `RNNStateManager`
- route/state 感知的 pinned-memory 加载
- 异步 remap 和 snapshot transfer 逻辑

这意味着当前 DTDG 的状态是：

- 已有统一接口
- 已有统一 kernel 协议
- 已有 Flare 风格流水线语义

但还不是 Flare 原版的高性能数据路径。

## 8. 配置系统

配置系统定义在：

- [default.yaml](/home/zlj/StarryUniGraph/starry_unigraph/config/default.yaml)
- [schema.py](/home/zlj/StarryUniGraph/starry_unigraph/config/schema.py)

### 8.1 主要配置段

- `model`
- `data`
- `train`
- `runtime`
- `sampling`
- `features`
- `graph`
- `dist`

### 8.2 graph mode 选择

`model.family` 会通过 `ModelRegistry` 解析到：

- `ctdg`
- `dtdg`

然后再驱动 provider 与 kernel 的选择。

### 8.3 校验行为

schema 层当前会：

- 合并默认配置
- 校验必填路径
- 自动推断 graph mode
- 对非激活字段发 warning
- 对关键缺失项抛出异常

## 9. 注册表系统

注册表的作用是把“名字”和“实现”解耦。

### 9.1 ModelRegistry

文件：

- [model_registry.py](/home/zlj/StarryUniGraph/starry_unigraph/registry/model_registry.py)

作用：

- 把 `model.name` 或 `model.family` 映射到 `ModelSpec`
- 确定 `graph_mode`

### 9.2 ProviderRegistry

文件：

- [provider_registry.py](/home/zlj/StarryUniGraph/starry_unigraph/registry/provider_registry.py)

作用：

- 将 `ctdg` 映射为 `CTDGProvider`
- 将 `dtdg` 映射为 `DTDGProvider`

### 9.3 TaskRegistry

文件：

- [task_registry.py](/home/zlj/StarryUniGraph/starry_unigraph/registry/task_registry.py)

作用：

- 解析任务适配器
- 让任务语义与图执行语义分离

## 10. Runtime 与 Checkpoint

运行时抽象定义在 [runtime/base.py](/home/zlj/StarryUniGraph/starry_unigraph/runtime/base.py)。

当前 provider 使用的轻量 runtime adapter 位于 [common.py](/home/zlj/StarryUniGraph/starry_unigraph/providers/common.py)。

checkpoint 读写位于 [io.py](/home/zlj/StarryUniGraph/starry_unigraph/checkpoint/io.py)。

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

CTDG 一般包括：

- `memory_state`
- `mailbox_state`
- `sampler_state`
- `executor_state`

DTDG 一般包括：

- `window_state`
- `snapshot_state`
- `route_cache`
- `executor_state`

## 11. 预处理层

预处理抽象定义在 [base.py](/home/zlj/StarryUniGraph/starry_unigraph/preprocess/base.py)。

当前 provider 预处理器主要负责：

- 创建产物目录
- 写 partition/route/index manifest
- 导出 artifact metadata

### 当前限制

目前预处理器还没有完成 BTS/Flare 原生级别的高性能预处理。

具体来说：

- CTDG 还没有完全打通真实事件排序与原生 sampler 索引生成
- DTDG 还没有完成 Flare `PartitionData` 级别的 snapshot/chunk/partition 生成

## 12. CLI

命令行入口位于 [main.py](/home/zlj/StarryUniGraph/starry_unigraph/cli/main.py)。

支持命令：

- `prepare`
- `train`
- `predict`
- `resume`

示例：

```bash
python -m starry_unigraph.cli prepare --config path/to/config.yaml
python -m starry_unigraph.cli train --config path/to/config.yaml
python -m starry_unigraph.cli predict --config path/to/config.yaml --split test
python -m starry_unigraph.cli resume --config path/to/config.yaml --checkpoint path/to/ckpt.pkl
```

## 13. 当前运行流程

### 13.1 创建 session

```python
from starry_unigraph import SchedulerSession

session = SchedulerSession.from_config("starry_unigraph/config/default.yaml")
```

### 13.2 预处理与构建

```python
session.prepare_data()
session.build_runtime()
```

### 13.3 训练

```python
summary = session.run_epoch(split="train")
```

### 13.4 预测

```python
result = session.predict(split="test")
```

### 13.5 保存与恢复

```python
session.save_checkpoint("checkpoint.pkl")
session.load_checkpoint("checkpoint.pkl")
```

## 14. 流水线语义

当前库的核心原则之一，是统一接口但不抹平原始执行语义。

### 14.1 CTDG 的语义

保留：

- 先采样
- 再取特征
- 再取状态
- 再做 memory updater
- 再做 sampled temporal neighborhood 上的 attention 聚合
- 再生成 message
- 再汇总为状态转移
- 再写回 memory/mailbox

trace 输出中会包含：

- `pipeline`
- `stage_payloads`
- `state`

### 14.2 DTDG 的语义

保留：

- 加载快照
- 应用 route
- 取时序状态
- 执行 snapshot propagation 和 temporal fusion
- 写回时序状态

trace 输出中会包含：

- `pipeline`
- `stage_payloads`
- `state`
- `spmm_output`
- `aggregated`

## 15. 当前已经从 BTS 和 Flare 合并了什么

### 已经合并的部分

- 统一 session/provider/config/checkpoint 架构
- BTS sampler 的源码与二进制资产
- BTS sampler 的 Python 包装与加载入口
- 统一 kernel protocol
- BTS 风格 CTDG pipeline 结构
- Flare 风格 DTDG pipeline 结构

### 还没有完整合并的部分

- BTS 原生 sampler 驱动的完整 CTDG 训练路径
- BTS 原生 memory/mailbox 关键路径
- Flare `PartitionData`
- Flare `STGraphLoader`
- Flare `RNNStateManager`
- Flare 高性能预处理与异步数据搬运路径

## 16. 测试覆盖

测试位于 [test_session.py](/home/zlj/StarryUniGraph/tests/test_session.py)。

当前覆盖内容包括：

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

当前环境说明：

- 已验证 `python -m compileall`
- 当前环境没有安装 `pytest`，所以测试文件已经写好，但未通过 `pytest` runner 执行

## 17. 推荐的后续工作

如果目标是“结构统一 + 性能也接近原系统”，推荐后续按下面顺序推进：

1. 让 `CTDGKernel` 的 batch 生成真正切换到 vendored BTS native sampler
2. 把 BTS 的 memory/mailbox 路径继续迁进仓库
3. 将 Flare `PartitionData` 合并进当前仓库
4. 将 Flare `STGraphLoader` 与状态管理器合并进当前仓库
5. 让 `DTDGKernel` 从当前模拟 snapshot 载入切换到 Flare 原生数据结构

## 18. 总结

当前 `StarryUniGraph` 已经具备：

- 一个统一的用户入口
- 一套统一的配置系统
- 一套统一的 checkpoint 格式
- 一套统一的执行协议
- 两套 family-specific kernel

它已经完成了“统一架构骨架”和“保留原流水线语义”的工作。接下来的重点不再是设计架构，而是把 BTS 和 Flare 的高性能后端逐步迁移并替换当前简化执行路径。
