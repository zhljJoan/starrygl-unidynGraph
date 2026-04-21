# 统一架构大纲：DTDG + CTDG 双模调度

---

## 总体分层

```
┌─────────────────────────────────────────────────────────────┐
│                      Application Layer                       │
│              Model Forward / Loss / Optimizer                │
│         (不感知 DTDG/CTDG, 只接收标准 tensor)               │
├─────────────────────────────────────────────────────────────┤
│                      Execution Layer                         │
│      Chunk Pipeline Engine (统一的流水线调度器)               │
│      ┌──────────────┐  ┌──────────────┐                     │
│      │  DTDG Runner  │  │  CTDG Runner  │                    │
│      │  (快照重组)    │  │  (事件流重组)  │                    │
│      └──────┬───────┘  └──────┬───────┘                     │
│             └────────┬────────┘                              │
│                      │                                       │
├──────────────────────┼──────────────────────────────────────┤
│                      │  Data Layer                           │
│      ┌───────────────┴───────────────┐                      │
│      │       ChunkAtomic Pool         │                      │
│      │   (统一的最小调度/数据单元)      │                      │
│      └───────────────┬───────────────┘                      │
│                      │                                       │
├──────────────────────┼──────────────────────────────────────┤
│                      │  State Layer                          │
│      ┌───────────────┴───────────────┐                      │
│      │      MemoryBank (CTDG)         │                      │
│      │      EmbeddingTable (DTDG)     │                      │
│      │      FeatureStore (共享)        │                      │
│      └───────────────┬───────────────┘                      │
│                      │                                       │
├──────────────────────┼──────────────────────────────────────┤
│                      │  Communication Layer                  │
│      ┌───────────────┴───────────────┐                      │
│      │      CommEngine (统一通信)      │                      │
│      │  spatial sync / state sync     │                      │
│      │  delta push / cache fetch      │                      │
│      └───────────────────────────────┘                      │
└─────────────────────────────────────────────────────────────┘
```

---

## 一、Data Layer：ChunkAtomic Pool

```
ChunkAtomic 是 DTDG 和 CTDG 共享的底层数据单元
不同模式下 chunk 的语义不同, 但结构统一

┌─────────────────────────────────────────────────────────┐
│  ChunkAtomic                                             │
│                                                          │
│  ┌── 通用字段 ─────────────────────────────────────────┐│
│  │  chunk_id: (time_slice, node_cluster)                ││
│  │  time_range: (t_start, t_end)                        ││
│  │  node_set: Tensor                                    ││
│  │  tcsr_rowptr / col / ts / edge_id                    ││
│  │  cross_node_ids / cross_node_home                    ││
│  │  load_estimate: float                                ││
│  └──────────────────────────────────────────────────────┘│
│                                                          │
│  ┌── 模式相关 (互斥, 由 chunk 生成器决定) ──────────────┐│
│  │  DTDG: 快照子图, 全部节点参与 GNN                    ││
│  │  CTDG: 事件流 (event_src, event_dst, event_time)     ││
│  │        + local_update_nodes                           ││
│  └──────────────────────────────────────────────────────┘│
│                                                          │
│  ┌── 统一接口 ─────────────────────────────────────────┐│
│  │  materialize(sampler_cfg) → (MFG, remote_manifest)   ││
│  │    DTDG: 全子图构建 MFG, 标记跨分区节点              ││
│  │    CTDG: 时序邻居采样 + 负采样, 构建 MFG             ││
│  │  两者产出格式相同: MFG + remote需求清单               ││
│  └──────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────┘

ChunkAtomic Pool:
  构建: 原始数据 → 时间切分 → 节点聚类 → chunk 生成
  分配: 按 (μ_i, σ_i²) 轮循分配到各设备
  重组: DTDG Runner / CTDG Runner 从 pool 中组合 chunk 序列
```

---

## 二、Execution Layer：双模 Runner + 统一 Pipeline Engine

```
┌─────────────────────────────────────────────────────────┐
│  Pipeline Engine (统一调度器)                             │
│                                                          │
│  职责:                                                   │
│    管理 compute / comm / cache 三条 CUDA stream          │
│    驱动 prepare → compute → update → step 四阶段循环     │
│    不关心 DTDG/CTDG 的具体语义                          │
│                                                          │
│  接口:                                                   │
│    engine.run(chunk_sequence, runner, state_manager)      │
│                                                          │
│  内部:                                                   │
│    for chunk in chunk_sequence:                          │
│      data = runner.prepare_data(chunk, state_manager)    │
│      [overlap: compute prev batch]                       │
│      runner.sync_data(data)                              │
│      output = runner.compute(data, model)                │
│      runner.post_compute(output, chunk, state_manager)   │
│      state_manager.step(next_chunk_idx)                  │
│                                                          │
└─────────────────────────────────────────────────────────┘

Runner 接口 (DTDG 和 CTDG 分别实现):

┌─────────────────────────────────────────────────────────┐
│  class Runner(Protocol):                                 │
│                                                          │
│    def build_sequence(pool, device_chunks) → chunk_seq   │
│      # 从 pool 中组合出执行序列                          │
│      # DTDG: 按时间排列快照, 随机选 cluster 子集         │
│      # CTDG: 按时间排列事件批次, 尽量多 cluster 并发     │
│                                                          │
│    def prepare_data(chunk, state_mgr) → BatchData        │
│      # 采样 + 预取状态                                   │
│      # DTDG: 全子图 + 预取跨分区 embedding               │
│      # CTDG: 时序采样 + 预取 memory                      │
│                                                          │
│    def sync_data(data) → None                            │
│      # 等异步预取完成                                     │
│      # DTDG: patch 远程 embedding                        │
│      # CTDG: patch 远程 memory                           │
│                                                          │
│    def compute(data, model) → output                     │
│      # forward + backward                                │
│      # 模型不感知 DTDG/CTDG                              │
│                                                          │
│    def post_compute(output, chunk, state_mgr) → None     │
│      # 更新状态 + 异步推送                                │
│      # DTDG: 更新 embedding + push 跨分区梯度/embedding  │
│      # CTDG: 更新 memory + push 增量                     │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### DTDG Runner 的 chunk 重组

```
DTDG: 长窗口时序依赖, 显存压力大

重组策略:
  每个时间步: 从 pool 中选取同一时间片的多个 cluster chunk
  显存限制: 不能全部加载 → 随机选子集
  跨时间步: 顺序执行 (时序依赖)

chunk_sequence = [
  [chunk(t=0,c=0), chunk(t=0,c=2).            ],  # 时间步0
  [chunk(t=1,c=0), chunk(t=1,c=2), chunk(t=1,c=5)],  # 时间步1
  ...
]

多设备: 各设备持有不同 cluster 子集, 数据并行
```

### CTDG Runner 的 chunk 重组

```
CTDG: 短窗口时序依赖, 显存压力小

重组策略:
  每个批次: 同一时间窗口内的事件, 可跨多个 cluster
  短窗口: 连续几个时间片的 chunk 组合
  可并发: 同时间内不同 cluster 的 chunk 无时序依赖

chunk_sequence = [
  chunk(t=0,c=0),  # batch 0
  chunk(t=0,c=1),  # batch 1 (与 batch 0 并发，全部partition并发直接合并成一个)
  chunk(t=1,c=0),  # batch 2 (依赖 batch 0)
  ...
]

多设备: 各设备持有不同 cluster, 数据并行
```

---

## 三、State Layer：统一的状态管理抽象

```
┌─────────────────────────────────────────────────────────┐
│  StateManager (Protocol)                                 │
│                                                          │
│  统一接口, DTDG 和 CTDG 各自实现:                        │
│                                                          │
│    prepare(needed_ids, query_context) → StateHandle      │
│      # 预取需要的节点状态到连续 buffer                    │
│      # DTDG: embedding gather + 跨分区 async fetch       │
│      # CTDG: memory gather (owned/hot/decay) + miss fetch│
│                                                          │
│    patch(handle) → None                                  │
│      # 等异步完成, 补全 miss                              │
│                                                          │
│    update(node_ids, new_state) → None                    │
│      # 写回 + 异步推送增量                                │
│      # DTDG: embedding 写回 + push                       │
│      # CTDG: memory 写回 + push delta                    │
│                                                          │
│    step(next_chunk_idx) → None                           │
│      # chunk 切换时的缓存维护                             │
│      # DTDG: 预加载下一快照的跨分区 embedding (可选)      │
│      # CTDG: 衰减缓存更新 (reuse_table 驱动)             │
│                                                          │
└─────────────────────────────────────────────────────────┘

两个实现:

┌── DTDGStateManager ───────────────────────────────────┐
│                                                        │
│  核心存储:                                              │
│    owned_embedding: [N_local, d]   本分区节点 embedding │
│    neighbor_embedding: [N_nbr, d]  邻居分区 embedding   │
│    (等价于 CTDG 的 owned_memory + cache)                │
│                                                        │
│  通信特点:                                              │
│    CommPlan 完全静态 (快照拓扑固定)                      │
│    每个快照的 send/recv 列表预编译                       │
│    all_to_all 通信                                      │
│                                                        │
│  缓存策略:                                              │
│    邻居分区 embedding 常驻 (快照间复用度高)              │
│    或按快照序列预加载 (类似 CTDG 的 step_cache)          │
│                                                        │
└────────────────────────────────────────────────────────┘

┌── CTDGStateManager (= MemoryBank) ────────────────────┐
│                                                        │
│  核心存储:                                              │
│    owned_memory + hot_cache + decay_cache               │
│    mailbox                                              │
│                                                        │
│  通信特点:                                              │
│    增量推送 (每 batch 推更新的节点)                      │
│    miss 按需拉取 (量化可选)                              │
│    CommPlan 半动态 (采样随机性)                          │
│                                                        │
│  缓存策略:                                              │
│    reuse_table 驱动衰减缓存                             │
│    热点常驻                                              │
│                                                        │
└────────────────────────────────────────────────────────┘
```

---

## 四、Communication Layer：统一通信引擎

```
┌─────────────────────────────────────────────────────────┐
│  CommEngine                                              │
│                                                          │
│  不区分 DTDG/CTDG, 只提供通信原语:                       │
│                                                          │
│  ┌── 空间通信 (Spatial) ──────────────────────────────┐ │
│  │  async_exchange(send_ids, send_vals, plan) → handle │ │
│  │    DTDG: 交换跨分区节点 embedding (静态 plan)       │ │
│  │    CTDG: 交换 miss 节点 memory (动态 plan)          │ │
│  │  实现: all_to_all 或 per-peer isend/irecv           │ │
│  └─────────────────────────────────────────────────────┘ │
│                                                          │
│  ┌── 状态同步 (State) ────────────────────────────────┐ │
│  │  async_push_delta(node_ids, values) → handle        │ │
│  │    DTDG: 推 embedding 更新                          │ │
│  │    CTDG: 推 memory 增量                             │ │
│  │  try_recv_delta() → Optional[(ids, values)]         │ │
│  └─────────────────────────────────────────────────────┘ │
│                                                          │
│  ┌── 缓存拉取 (Cache Fetch) ─────────────────────────┐ │
│  │  async_fetch(node_ids, owners) → handle             │ │
│  │    step_cache 时拉取新缓存数据                       │ │
│  │    可选量化 (INT8)                                   │ │
│  └─────────────────────────────────────────────────────┘ │
│                                                          │
│  ┌── 梯度同步 (共享) ────────────────────────────────┐  │
│  │  all_reduce_gradients()                             │ │
│  │    DTDG/CTDG 共用                                   │ │
│  └─────────────────────────────────────────────────────┘ │
│                                                          │
│  所有通信在 comm_stream 上执行                           │
│  与 compute_stream 天然重叠                              │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

## 五、端到端流程对比

```
DTDG 流程:

  Pool ──► DTDGRunner.build_sequence()
           ┌─────────────────────────────────────┐
           │ for snapshot_t:                      │
           │   chunks = select_clusters(t)        │
           │   mfg = chunk.materialize(full_graph)│
           │   state = dtdg_state.prepare(mfg)    │
           │   [overlap: compute prev snapshot]   │
           │   dtdg_state.patch(state)            │
           │   out = model(mfg, state)            │
           │   dtdg_state.update(out)             │
           │   dtdg_state.step(t+1)               │
           └─────────────────────────────────────┘

CTDG 流程:

  Pool ──► CTDGRunner.build_sequence()
           ┌─────────────────────────────────────┐
           │ for chunk_i:                         │
           │   mfg = chunk.materialize(sample)    │
           │   state = ctdg_state.prepare(mfg)    │
           │   [overlap: compute prev batch]      │
           │   ctdg_state.patch(state)            │
           │   out = model(mfg, state)            │
           │   ctdg_state.update(out)             │
           │   ctdg_state.step(i+1)               │
           └─────────────────────────────────────┘

结构完全相同! 只有 Runner 和 StateManager 的实现不同
PipelineEngine 是同一份代码
```

---

## 六、模块依赖关系

```
┌─────────────────────────────────────────────────────┐
│                    User Code                         │
│  cfg = load_config("dtdg" | "ctdg")                 │
│  engine.run(cfg, model, data)                        │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│              PipelineEngine                          │
│  run(chunk_seq, runner, state_mgr, comm, model)      │
└──┬──────────┬──────────┬──────────┬─────────────────┘
   │          │          │          │
   ▼          ▼          ▼          ▼
┌──────┐ ┌────────┐ ┌────────┐ ┌────────┐
│Runner│ │StateMgr│ │CommEng │ │  Model │
│(接口)│ │(接口)  │ │(共享)  │ │(共享)  │
└──┬───┘ └──┬─────┘ └────────┘ └────────┘
   │        │
   ▼        ▼
┌──────────────────┐  ┌──────────────────┐
│  DTDGRunner      │  │  CTDGRunner      │
│  DTDGStateMgr    │  │  CTDGStateMgr    │
│                  │  │  (= MemoryBank)  │
└──────┬───────────┘  └──────┬───────────┘
       │                     │
       └──────────┬──────────┘
                  ▼
        ┌──────────────────┐
        │  ChunkAtomic Pool │
        │  FeatureStore     │
        │  GlobalCSR        │
        └──────────────────┘
```

---

## 七、关键统一点与差异点

```
┌──────────────────┬────────────────────┬────────────────────┐
│                  │      DTDG          │      CTDG          │
├──────────────────┼────────────────────┼────────────────────┤
│ ChunkAtomic      │     共享结构                            │
│ Pool构建         │     共享 (时间切分 + 节点聚类)           │
│ 设备分配         │     共享 (μ,σ 轮循)                     │
├──────────────────┼────────────────────┼────────────────────┤
│ materialize      │  全子图 MFG        │  采样 MFG          │
│ 重组策略         │  同时间多cluster   │  连续时间多cluster  │
│ 状态类型         │  embedding         │  memory + mail     │
│ 通信模式         │  静态 CommPlan     │  动态 miss + delta │
│ 缓存策略         │  邻居分区常驻      │  hot+decay+reuse   │
├──────────────────┼────────────────────┼────────────────────┤
│ PipelineEngine   │     共享                                │
│ CommEngine       │     共享                                │
│ Runner接口       │     共享 (Protocol)                     │
│ StateManager接口 │     共享 (Protocol)                     │
│ Model接口        │     共享                                │
└──────────────────┴────────────────────┴────────────────────┘
```

---

## 八、构建与初始化流程

```
┌─────────────────────────────────────────────────────────┐
│  Phase 0: 数据加载与 Chunk 构建 (一次性, 离线)           │
│                                                          │
│  raw_data ──► 时间切分 ──► 节点聚类 ──► ChunkAtomic Pool │
│  同时构建: GlobalCSR, FeatureStore                       │
│  模式无关: DTDG/CTDG 共享同一个 Pool                     │
│                                                          │
├─────────────────────────────────────────────────────────┤
│  Phase 1: 设备分配 (一次性)                               │
│                                                          │
│  Pool ──► (μ_i, σ_i²) 排序轮循 ──► device assignment    │
│  模式无关                                                │
│                                                          │
├─────────────────────────────────────────────────────────┤
│  Phase 2: 模式特化初始化                                  │
│                                                          │
│  if DTDG:                                                │
│    构建 DTDGStateMgr (embedding表 + 邻居分区缓存)        │
│    构建 DTDGRunner (快照重组策略)                         │
│    预编译 CommPlan (静态)                                 │
│                                                          │
│  if CTDG:                                                │
│    构建 CTDGStateMgr (memory三层 + mailbox)              │
│    计算 hot_nodes (图结构分析)                            │
│    预计算 reuse_table (chunk序列分析)                     │
│    构建 CTDGRunner (事件流重组策略)                       │
│                                                          │
├─────────────────────────────────────────────────────────┤
│  Phase 3: 训练循环                                        │
│                                                          │
│  chunk_seq = runner.build_sequence(pool)                  │
│  engine.run(chunk_seq, runner, state_mgr, comm, model)   │
│  # 统一的 Pipeline Engine 驱动, 模式无关                  │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

## 九、代码组织

```
starrygl/
├── data/
│   ├── chunk_atomic.py          # ChunkAtomic 定义
│   ├── chunk_builder.py         # 时间切分 + 节点聚类 → Pool
│   ├── feature_store.py         # 节点/边特征分片存储
│   └── global_csr.py            # 全图 CSR
│
├── runtime/
│   ├── engine.py                # PipelineEngine (统一)
│   │
│   ├── runner/
│   │   ├── base.py              # Runner Protocol
│   │   ├── dtdg_runner.py       # DTDG: 快照重组 + 全图 materialize
│   │   └── ctdg_runner.py       # CTDG: 事件流重组 + 采样 materialize
│   │
│   ├── state/
│   │   ├── base.py              # StateManager Protocol
│   │   ├── rnn_state.py           # RNNtateMgr (embedding + 邻居缓存)
│   │   └── memory_state.py        # CTDGStateMgr (MemoryBank)
│   │
│   ├── comm/
│   │   ├── comm_engine.py       # CommEngine (统一通信原语)
│   │   ├── spatial_plan.py      # SpatialDeps (空间依赖)
│   │   └── state_plan.py        # StateDeps (状态依赖)
│   │
│   └── cache/
│       ├── hot_cache.py         # 热点常驻缓存
│       ├── decay_cache.py       # 衰减缓存 + reuse_table
│       └── partition_cache.py   # 邻居分区缓存 (DTDG侧)
│
├── csrc/                        # C++ / CUDA 后端
│   ├── memory_backend.cpp       # prepare / patch / update / step_cache
│   ├── sampler.cpp              # 时序采样 / 全图采样
│   ├── quantize.cu              # INT8 量化/反量化
│   └── comm_ops.cpp             # 通信原语封装
│
└── models/
    ├── tgn.py                   # TGN (CTDG)
    ├── dysat.py                 # DySAT (DTDG)
    └── ...
```

---

## 十、统一入口

```python
def train(cfg):
    # ---- 模式无关 ----
    pool = ChunkBuilder.build(cfg.data, cfg.time_slices, cfg.node_clusters)
    assignment = assign_devices(pool, cfg.world_size)
    global_csr = GlobalCSR.load(cfg.data)
    features = FeatureStore.load(cfg.data, assignment)
    comm = CommEngine(cfg.dist_ctx)

    # ---- 模式特化 ----
    if cfg.mode == "dtdg":
        runner = DTDGRunner(pool, assignment, cfg)
        state_mgr = DTDGStateMgr(assignment, features, comm, cfg)
    elif cfg.mode == "ctdg":
        runner = CTDGRunner(pool, assignment, cfg)
        state_mgr = CTDGStateMgr(assignment, features, comm, cfg)

    model = build_model(cfg.model)
    engine = PipelineEngine(compute_stream, comm_stream, cache_stream)

    # ---- 统一训练 ----
    for epoch in range(cfg.epochs):
        chunk_seq = runner.build_sequence(pool, epoch)
        engine.run(chunk_seq, runner, state_mgr, comm, model)
```