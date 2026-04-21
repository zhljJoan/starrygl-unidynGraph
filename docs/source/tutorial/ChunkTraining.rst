Chunk 流水线训练指南
====================

Chunk 路径是 StarryUniGraph 的下一代统一训练路径，以 **ChunkAtomic** 为底层数据单元，
通过 **PipelineEngine** 统一调度 DTDG 和 CTDG 两种模式的训练流程。

.. note::

    Chunk 路径当前处于框架就绪状态（``raise NotImplementedError``），
    计算逻辑待实现。可用于架构参考和接口设计验证。

设计目标
--------

.. code-block:: text

    ┌─────────────────────────────────────────────────────────┐
    │                      PipelineEngine                      │
    │         (统一调度器：DTDG + CTDG 共用同一份代码)          │
    │                                                          │
    │  for chunk in chunk_seq:                                 │
    │    data   = runner.prepare_data(chunk, state_mgr)        │
    │    [overlap: compute prev batch]                         │
    │    runner.sync_data(data)                                │
    │    output = runner.compute(data, model)                  │
    │    runner.post_compute(output, chunk, state_mgr)         │
    │    state_mgr.step(next_chunk_idx)                        │
    │                                                          │
    └───────────────────────────┬─────────────────────────────┘
                                │
               ┌────────────────┴──────────────────┐
               ▼                                   ▼
        DTDGRunner                          CTDGRunner
        DTDGStateManager                    CTDGStateManager
        (快照重组 + embedding)              (事件流重组 + memory)

与现有路径的关系
----------------

.. list-table::
   :header-rows: 1
   :widths: 20 25 25 30

   * - 特性
     - CTDG 路径
     - DTDG 路径
     - Chunk 路径（目标）
   * - 数据单元
     - Event batch
     - STGraphBlob
     - ChunkAtomic
   * - 状态管理
     - CTDGMemoryBank
     - RNNStateManager
     - StateManager Protocol
   * - 调度器
     - CTDGOnlineRuntime
     - FlareRuntimeLoader
     - PipelineEngine（统一）
   * - 通信
     - 按需 fetch
     - all_to_all（静态）
     - CommEngine（统一原语）
   * - 实现状态
     - ✅ 完整
     - ✅ 完整
     - 🔧 框架就绪

快速开始（框架验证）
--------------------

**预处理（数据切分为 ChunkAtomic Pool）：**

.. code-block:: bash

    python train.py --mode chunk --phase prepare \
        --config configs/chunk_default.yaml

**训练（待实现后可用）：**

.. code-block:: bash

    torchrun --nproc_per_node=4 train.py --mode chunk --phase train

**导入验证（无需完整运行时）：**

.. code-block:: python

    from starry_unigraph.data.chunk_atomic import ChunkAtomic
    from starry_unigraph.runtime.engine import PipelineEngine
    from starry_unigraph.runtime.runner.base import Runner
    from starry_unigraph.runtime.state.base import StateManager
    from starry_unigraph.runtime.comm.comm_engine import CommEngine

    engine = PipelineEngine(overlap=True)
    print(engine.compute_stream)   # 默认使用当前 CUDA stream

ChunkAtomic 数据结构
--------------------

ChunkAtomic 是 Chunk 路径的最小调度/数据单元，对应一个时间切片 × 节点簇：

.. code-block:: python

    from starry_unigraph.data.chunk_atomic import ChunkAtomic
    import torch

    chunk = ChunkAtomic(
        chunk_id=(0, 2),                       # (time_slice_id=0, cluster_id=2)
        time_range=(1000.0, 2000.0),           # 时间范围
        node_set=torch.tensor([10, 20, 30]),   # 本 cluster 的 master 节点 ID
        tcsr_rowptr=torch.tensor([0, 2, 4, 5]),
        tcsr_col=torch.tensor([20, 30, 10, 30, 10]),
        tcsr_ts=torch.tensor([1100., 1200., 1300., 1400., 1500.]),
        tcsr_edge_id=torch.tensor([0, 1, 2, 3, 4]),
        cross_node_ids=torch.tensor([50, 60]),  # 跨界邻居
        cross_node_home=torch.tensor([3, 3]),   # 归属 cluster
        cross_edge_count=torch.zeros(8),
        load_estimate=5.0,
    )

    # materialize 是核心接口（待 C++ 实现）
    # local_mfg, remote_manifest = chunk.materialize(sampler_cfg)

ChunkBuilder：构建 Chunk Pool
------------------------------

.. code-block:: python

    from starry_unigraph.data.chunk_builder import ChunkBuilder, ChunkBuildConfig

    cfg = ChunkBuildConfig(
        num_time_slices=100,    # 时间轴切分数
        num_clusters=8,         # 每个时间片的节点簇数
        cluster_method="metis", # 聚类算法
        adaptive_split=False,   # 均匀切分
    )

    builder = ChunkBuilder()
    pool = builder.build(cfg, raw_graph)          # 生成 ChunkAtomic 列表
    assignment = builder.assign_devices(pool, 4)  # 分配到 4 个 rank

    # assignment[0] = 分配给 rank 0 的 chunk 列表
    # 按 load_estimate 贪心均衡分配

PipelineEngine 执行流程
-----------------------

.. code-block:: python

    from starry_unigraph.runtime.engine import PipelineEngine
    from starry_unigraph.runtime.runner.dtdg_runner import DTDGRunner
    from starry_unigraph.runtime.state.rnn_state import DTDGStateManager
    import torch

    # 创建三条 CUDA stream
    compute_stream = torch.cuda.Stream()
    comm_stream    = torch.cuda.Stream()
    cache_stream   = torch.cuda.Stream()

    engine = PipelineEngine(
        compute_stream=compute_stream,
        comm_stream=comm_stream,
        cache_stream=cache_stream,
        overlap=True,           # 启用 double-buffering（通信与计算重叠）
    )

    runner    = DTDGRunner()
    state_mgr = DTDGStateManager(num_local_nodes=1000, feat_dim=64,
                                  world_size=4, device=device, comm=comm)

    chunk_seq = runner.build_sequence(pool, assignment[rank])

    # 训练循环（PipelineEngine 统一驱动）
    engine.run(chunk_seq, runner, state_mgr, comm, model)

Runner Protocol 实现
--------------------

实现自定义 Runner 只需继承并实现 5 个方法：

.. code-block:: python

    from starry_unigraph.runtime.runner.base import Runner, BatchData
    from starry_unigraph.data.chunk_atomic import ChunkAtomic
    from starry_unigraph.runtime.state.base import StateManager
    from typing import Any, Sequence

    class MyRunner:
        def build_sequence(self, pool, device_chunks) -> list[ChunkAtomic]:
            # 按业务逻辑排序 chunk（如时序、随机等）
            return sorted(device_chunks, key=lambda c: c.chunk_id)

        def prepare_data(self, chunk: ChunkAtomic, state_mgr: StateManager) -> BatchData:
            # 采样 + 异步预取状态
            # local_mfg, manifest = chunk.materialize(sampler_cfg)
            # handle = state_mgr.prepare(local_mfg.node_ids, chunk.spatial_deps)
            raise NotImplementedError

        def sync_data(self, data: BatchData) -> None:
            # state_mgr.patch(data.state_handle)
            raise NotImplementedError

        def compute(self, data: BatchData, model: Any) -> Any:
            # output = model(data.mfg, data.state_handle.gathered_states)
            # loss = criterion(output, labels)
            # loss.backward()
            raise NotImplementedError

        def post_compute(self, output: Any, chunk: ChunkAtomic, state_mgr: StateManager) -> None:
            # state_mgr.update(output.node_ids, output.new_state)
            raise NotImplementedError

StateManager Protocol 实现
---------------------------

.. code-block:: python

    from starry_unigraph.runtime.state.base import StateManager, StateHandle
    from torch import Tensor
    from typing import Any

    class MyStateManager:
        def prepare(self, needed_ids: Tensor, query_context: Any) -> StateHandle:
            # 三层缓存查找 + 异步 fetch miss 节点
            raise NotImplementedError

        def patch(self, handle: StateHandle) -> None:
            # handle.comm_handle.wait()  等待 fetch 完成
            raise NotImplementedError

        def update(self, node_ids: Tensor, new_state: Tensor) -> None:
            # 写回 + 推送增量
            raise NotImplementedError

        def step(self, next_chunk_idx: int) -> None:
            # 缓存维护（衰减、预加载）
            raise NotImplementedError

CommEngine Protocol
-------------------

.. code-block:: python

    from starry_unigraph.runtime.comm.comm_engine import CommEngine
    from starry_unigraph.runtime.comm.spatial_plan import SpatialDeps

    class MyCommEngine:
        def async_exchange(self, send_ids, send_vals, plan: SpatialDeps):
            # all_to_all or isend/irecv
            ...

        def async_push_delta(self, node_ids, values):
            # 向 owner rank 推送增量
            ...

        def try_recv_delta(self):
            # 非阻塞查询是否有增量到达
            ...

        def async_fetch(self, node_ids, owners):
            # step_cache 预加载
            ...

        def all_reduce_gradients(self):
            # DDP all-reduce（或 no-op）
            ...

缓存系统
--------

Chunk 路径内置三种缓存，由 ``StateManager`` 管理：

.. code-block:: python

    from starry_unigraph.runtime.cache.hot_cache import HotCache
    from starry_unigraph.runtime.cache.decay_cache import DecayCache
    from starry_unigraph.runtime.cache.partition_cache import PartitionCache

    # CTDG 侧：三层 memory 缓存
    hot_cache   = HotCache(capacity=10000, feat_dim=64, device=device)
    decay_cache = DecayCache(capacity=50000, feat_dim=64, device=device,
                              reuse_table=precomputed_reuse_table)

    # DTDG 侧：邻居分区 embedding 缓存
    part_cache  = PartitionCache(world_size=4, feat_dim=64, device=device)

    # 使用方式（待实现）
    hit_mask, cached_vals = hot_cache.lookup(node_ids)
    hot_cache.update(node_ids, new_vals)
    evicted = hot_cache.evict(threshold=5)

当前实现状态
------------

.. list-table::
   :header-rows: 1
   :widths: 40 20 40

   * - 组件
     - 状态
     - 说明
   * - ``ChunkAtomic`` 数据结构
     - ✅ 完成
     - 字段定义、接口签名完整
   * - ``ChunkBuilder``
     - 🔧 骨架
     - ``build()`` / ``assign_devices()`` 待实现
   * - ``PipelineEngine``
     - 🔧 骨架
     - ``run()`` / ``run_eval()`` 待实现
   * - ``DTDGRunner`` / ``CTDGRunner``
     - 🔧 骨架
     - 5 个方法均待实现
   * - ``DTDGStateManager``
     - 🔧 骨架
     - 4 个方法均待实现
   * - ``CTDGStateManager``
     - 🔧 骨架
     - 4 个方法均待实现
   * - ``CommEngine``
     - Protocol 定义
     - 需提供具体实现类
   * - ``HotCache`` / ``DecayCache``
     - 🔧 骨架
     - lookup/update/evict 待实现
   * - ``PartitionCache``
     - 🔧 骨架
     - lookup/update/evict 待实现
   * - ``ChunkPreprocessor``
     - 🔧 骨架
     - 3 个方法均待实现
   * - ``ChunkRuntimeLoader``
     - 🔧 骨架
     - iter_* / run_*_step 待实现
