# Development Environment Notes

## Codex GPU Access

In this project, Codex commands run in a restricted sandbox by default. The
default sandbox can execute normal CPU/file-system commands, but it may not see
GPU device nodes such as `/dev/nvidia*`.

Observed on 2026-05-18:

- Default sandbox:
  - `ls -l /dev/nvidia*` produced no visible devices.
  - `nvidia-smi` failed with: `couldn't communicate with the NVIDIA driver`.
  - The effective groups were `zlj nogroup`.
- Escalated command execution:
  - `nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader`
    reported four NVIDIA A40 GPUs.

Conclusion: a default-sandbox `nvidia-smi` failure in Codex does not mean the
machine GPU driver is broken. It means the current Codex command sandbox does
not expose GPU devices.

For GPU-dependent validation, run the command with escalated permissions, for
example:

```text
nvidia-smi ...
python ...  # when the script initializes CUDA
torchrun --nproc_per_node=4 ...
```

When reporting test status, distinguish these two cases:

- "GPU unavailable in default Codex sandbox" means the sandbox lacks device
  access.
- "GPU/driver unavailable on host" should only be reported after an escalated
  command also fails.

# Chunk 链路预测任务异步通信改造设计

## 设计原则

`backends/chunk` 作为后续主链路，统一承载 CTDG 采样式 edge predict、node predict，以及 DTDG 全图/快照式 edge predict、node predict。核心原则：

- time-slice 先构建，runtime 不再切时间窗口。
- chunk 是节点集合；edge predict 的 event 按 `event_owner` 归属到计算 rank，node predict 的 target node 按 `target_owner` 归属到计算 rank。
- 索引参考 `DistIndex` packed 设计，runtime 热路径不维护冗余的 `global_id + owner + local_id` 三元组。
- DTDG 模型层的 `ChunkPropagationRoute` 在 prepare 阶段预构建；CTDG 的 `SpatialRouteData` 由采样输出驱动构建/缓存，不作为主预构建对象。
- 模型层从 backend 中抽离，单独目录管理。
- prepare 阶段可以使用 `argsort/unique/scatter/bincount`；train step 热路径减少 Python loop 和零碎 torch 调用。

## Time Slice

第一步先生成 `time_ptr: torch.int64[T + 1]`：

- CTDG：使用库内自适应切分，根据 event 时间密度、目标 batch size、采样负载估计生成 slice。
- DTDG：使用离散快照或固定窗口分区；如果输入已是 snapshot，`time_ptr` 直接指向 snapshot 边区间。
- 所有后续结构都以 `time_slice_id` 为第一维或外层 list：chunk load、event owner、spatial route、memory route、train/test split。

event 区间：

```text
slice t events = [time_ptr[t], time_ptr[t + 1])
```

## Chunk 划分

chunk 是节点集合。每个节点有且仅有一个 master chunk：

```text
num_chunks = num_partitions * chunks_per_partition

node_to_chunk              torch.int32/int64[num_nodes]
chunk_ptr                  torch.int64[num_chunks + 1]
chunk_nodes                torch.int64[num_nodes]
chunk_to_initial_partition torch.int16/int32[num_chunks]
chunk_to_owner_partition   torch.int16/int32[num_chunks]
```

`chunk_ptr/chunk_nodes` 是 `chunk_to_nodes` 的 CSR 形式，避免 Python list 在 runtime 中参与计算。`chunk_to_owner_partition` 可以在 trial/epoch 边界重平衡，但 step 内不变。

### 分区算法

支持四类策略：

- `metis`：先 node partition，再在每个 partition 内切 `chunks_per_partition` 个 chunk。
- `mem_share`：先按热点邻接亲和分配冷节点，再在 partition 内切 chunk。
- `speed_ctdg`：复用 CTDG/SPEED 流式分区结果，再派生 chunk。
- `chunk_metis_balance`：先切 `num_partitions * chunks_per_partition` 个 chunk，再按每个 chunk 的 time-slice 负载向量做均衡划分。

`chunk_metis_balance` 是需要补上的关键算法：

```text
1. 构建 time_ptr
2. 全图切 C = P * K 个 chunk，得到 node_to_chunk
3. 计算 L[t, c]，表示 chunk c 在 time-slice t 的计算负载
4. 构建 chunk graph:
   node weight = L[:, c]
   edge weight = chunk 间跨边、共享 remote 节点、同 slice 共现次数
5. 将 chunk graph 划分到 P 个 rank:
   minimize max_t max_p sum_{c owner p} L[t, c] + lambda * cut_cost
6. node_owner[node] = chunk_to_owner_partition[node_to_chunk[node]]
```

第一版实现可以用当前 `greedy_rebalance_by_slice`：维护 `[num_partitions, num_slices]` 负载矩阵，每次把一个 chunk 迁移到能降低最大 slice 峰值的 rank。后续替换成真正支持多维 node weight 的图划分器。

### Chunk Load 向量

`L[t, c]` 推荐由以下分量组成：

```text
edge_count[t,c]
remote_edge_count[t,c]
active_node_count[t,c]
remote_node_count[t,c]
sampling_load[t,c]
negative_fetch_load[t,c]
```

默认标量：

```text
L[t,c] = edge_count
       + 2.0 * remote_edge_count
       + sampling_weight * sampling_load
       + neg_weight * negative_fetch_load
```

实现上用 slice 内 edge tensor 做 `bincount/scatter_add`，禁止 edge 级 Python loop。

## Packed DistIndex

索引采用当前 `starry_unigraph/backends/chunk/data/dist_index.py` 的编码：

```text
bits 0..47    local row id
bit  48       shared/hot flag
bit  49       cached/local-readable flag
bits 50..65   partition id
```

只维护两张主索引：

```text
master_dist_index: torch.int64[num_nodes]
    loc    = node 在 master/owner store 的 local row
    part   = node_master 或 node_owner
    shared = replica_mask[node]
    cached = false

read_dist_index: torch.int64[num_nodes]
    loc    = node 在当前 rank 可读 store 的 row
    part   = 当前可读副本所在 rank；不可读时回退到 master part
    shared = replica_mask[node]
    cached = 本 rank 是否可直接读
```

使用规则：

- 权威写入、memory reduce、checkpoint 使用 `master_dist_index`。
- feature/memory 读取先查 `read_dist_index.cached`；命中本地 gather，未命中按 `master_dist_index.part` 发 fetch。
- route 中传 packed index，尽量不再同时传 `global_node_id`、`owner`、`local_id`。
- `node_owner/node_master` 保留在 prepare 和 manifest，用于构建 packed index 与调试，不进入每 step 热路径。

## Placement 与热点

prepare 输出：

```text
node_to_chunk          torch.int32/int64[num_nodes]
node_owner             torch.int16/int32[num_nodes]
node_master            torch.int16/int32[num_nodes]
replica_mask           torch.bool[num_nodes]
hot_replica_ptr        torch.int64[num_hot + 1]
hot_replica_rank       torch.int16/int32[num_replica_edges]
master_dist_index      torch.int64[num_nodes]
read_dist_index        torch.int64[num_nodes] per rank
placement_version      int
```

热点节点由 train + validation 范围内的全局度数/事件频率最高节点确定，避免 test 信息泄漏。热点可复制到多个 rank，本地读走 `read_dist_index.cached`，写回仍走 master。

edge predict 正样本 event `(src, dst, ts, eid)` 的计算归属：

```text
if src_hot xor dst_hot:
    event_owner = owner(non_hot_node)
elif not src_hot and not dst_hot:
    event_owner = owner(dst)
else:
    event_owner = master(dst) 或负载最低的热点副本 rank
```

这样“热点数据和非热点数据的连边”落到非热点所在分区，减少热点 master 写压力。

node predict 的 target node 计算归属：

```text
if target is hot:
    target_owner = node_master[target] 或持有最新 readable replica 的 rank
else:
    target_owner = node_owner[target]
```

CTDG node predict 的 target 可以来自事件触发节点、时间窗内活跃节点或显式 label node；DTDG node predict 的 target 来自 snapshot/window 的 local labeled nodes。target owner 只决定 loss/metric 在哪个 rank 计算；feature/memory 的读取仍由 packed `read_dist_index/master_dist_index` 决定。

## 数据结构

### Canonical Event Store

全局列式 event store：

```text
event_src        torch.int64[E]
event_dst        torch.int64[E]
event_ts         torch.int64/float32[E]
event_eid        torch.int64[E]
event_owner      torch.int16/int32[E]
event_dst_chunk  torch.int32/int64[E]
event_split      torch.uint8[E]      # 0=train, 1=val, 2=test
time_ptr         torch.int64[T + 1]
```

node predict 额外维护 target store：

```text
target_node      torch.int64[N_target]
target_ts        torch.int64/float32[N_target] optional for CTDG
target_label     task-specific tensor
target_owner     torch.int16/int32[N_target]
target_split     torch.uint8[N_target]
target_ptr       torch.int64[T + 1]  # slice/window -> target range
```

edge predict 使用 event store；node predict 使用 target store。两者共享 `time_ptr`、chunk placement、packed DistIndex 和 route 体系。

排序键：

```text
(time_slice, event_owner, dst_chunk, dst, ts, eid)
```

### PartitionData

chunk-native `PartitionData` 是磁盘/内存中的“分区图数据容器”，不是 runtime 的采样结果，也不是 DTDG snapshot index。它保存本 rank 在所有 time-slice/snapshot 下的 compact CSR 边结构和必要的节点列表，用于快速 materialize CTDG roots 或 DTDG block。

```text
src_ids    TensorData[int64]       # remote/read src global id 或 packed DistIndex
dst_ids    TensorData[int64]       # local dst global id
edge_ids   TensorData[int64]
edge_src   TensorData[int32/int64] # compact src row
edge_dst   TensorData[int32/int64] # compact dst row/sentinel
edge_ptr   TensorData[int64]
dst_chunk  TensorData[int32/int64]
edge_ts    TensorData[int64/float32] 或 edge_data["ts"]
```

约束：

- block/MFG 内部使用 compact row id。
- 跨 rank 使用 packed DistIndex。
- global node id 只在 canonical store、debug、task root 中保留。
- 不保存重复 owner/local id tensor。

### PartitionData 与 DTDG Snapshot Index 的边界

二者分工不同：

```text
PartitionData:
  粒度：rank 级 artifact，覆盖多个 time-slice/snapshot
  内容：packed TensorData、edge CSR、dst_chunk、edge_ts、node/edge feature 引用
  用途：loader 的底层数据源，负责按 slice 切出原始 compact 图数据

DTDG Snapshot Index:
  粒度：单个 snapshot/window 的 runtime view
  内容：dst compact rows、src=[dst, remote_src] compact rows、edge compact rows、block.route
  用途：直接喂给 DTDG encoder/head，包含本 snapshot 的通信边界和 id_map
```

因此 `snapshot_index_{rank}.pth` 不应复制完整 `PartitionData`。它只保存从 `PartitionData` 切片后能快速构造 block 的轻量索引，例如 `snapshot_ptr`、`src_row_ptr`、`dst_row_ptr`、`route_id`、`id_map_ptr`。

### CTDG Temporal Index

CTDG 采样索引用于 edge predict 和 node predict。edge predict 的 roots 来自 `pos_src/pos_dst/neg_dst`；node predict 的 roots 来自 `target_node` 以及模型需要的上下文节点。

```text
temporal_indptr       torch.int64[num_nodes + 1]
temporal_indices      torch.int64[num_edges]
temporal_eids         torch.int64[num_edges]
temporal_ts           torch.int64/float32[num_edges]
time_ptr              torch.int64[T + 1]
master_dist_index     torch.int64[num_nodes]
read_dist_index       torch.int64[num_nodes] per rank
```

### MemShareEventEngine 阶段与输出

`MemShareEventEngine` 属于 CTDG runtime 的 sample/materialize 阶段，位置在 task roots 已确定之后、feature/memory fetch 之前：

```text
task roots/pos-neg/targets
  -> MemShareEventEngine.sample()
  -> CTDGSampleResult
  -> build/merge fetch route
  -> CommPipeline.fetch feature/memory
  -> materialize BatchData/MFG
  -> model forward
```

它不是 prepare 阶段组件，不负责切 time-slice，也不负责跨 rank all-to-all。prepare 只提前构建 temporal index、time_ptr 和 route template。

`MemShareEventEngine` 只接收 contiguous tensor。采样输出 `CTDGSampleResult` 应包含：

```text
mfgs/native_blocks       backend-native sampled blocks，尚未 patch 远端 feature/memory
input_nodes              torch.int64[S] compact/global/packed，按 builder 约定
output_nodes             torch.int64[D] roots/targets
edge_ids                 torch.int64[E_sampled]
node_ts                  torch.int64/float32[S] optional
edge_ts                  torch.int64/float32[E_sampled] optional
remote_read_index        torch.int64[R] packed DistIndex, sorted by part
local_read_index         torch.int64[L] packed DistIndex or compact rows
id_map                   optional compact map for head/model
```

`remote_read_index` 是后续 fetch route 的输入；`mfgs/native_blocks` 是模型计算图的结构输入。二者不要混在一个“route”概念里。

### DTDG Snapshot Index

DTDG 使用 snapshot/window block：

```text
dst_ids local compact rows
src_ids = [dst_ids, remote_src_ids]
edge_src/edge_dst compact row ids
route: ChunkPropagationRoute
```

`ChunkPropagationRoute` 需要支持：

```text
append_recv=True  -> forward returns cat([local_x, recv_x])
append_recv=False -> forward returns recv_x only
```

backward 通过 autograd all-to-all 把 remote gradient 回传 owner，并对 `send_index` 做 `index_add_`。

### ChunkPropagationRoute 与 SpatialRouteData 的边界

二者不是同一个 route：

```text
SpatialRouteData:
  阶段：model forward 之前的 data fetch / feature fetch
  语义：把远端 feature/memory 数据搬到本 rank
  autograd：不承载 activation gradient；通常 no_grad 或作为输入数据
  输出：recv feature/memory tensor，patch 到 BatchData/MFG

ChunkPropagationRoute:
  阶段：DTDG 模型层内部
  语义：在 GNN layer / edge head 里交换 activation
  autograd：必须嵌入 torch.autograd.Function，backward 反向 all-to-all gradient
  输出：append_recv 时返回 [local_activation, remote_activation]
```

CTDG 采样链路主要用 `SpatialRouteData` 做输入数据搬运；DTDG 全图链路主要用 `ChunkPropagationRoute` 在模型层里做 activation 通信。二者可以共享 send/recv ptr 的构建思路，但不能共享同一个 runtime 对象。

## Route 构建与预构建

### DTDG 预构建 ChunkPropagationRoute

真正需要 prepare 阶段预构建的是 DTDG 模型层的 `ChunkPropagationRoute`。它按 `(rank, snapshot/window, layer 或 block)` 构建，直接挂到 DTDG block 或 `PropagationPlan` 上，供模型层 `RouteSendFunction/RouteRecvFunction` 使用：

```text
propagation_routes[rank][snapshot_id][layer_id]:
  send_index      torch.int64[S]   # local activation rows to send
  send_sizes      list[int] / int64[P]
  recv_sizes      list[int] / int64[P]
  recv_src_rows   optional int64[R] # remote rows appended after local dst rows
  append_recv     bool
```

构建输入来自 DTDG snapshot index：

```text
dst_ids for this rank
remote src ids required by edges into dst_ids
node owner/master packed index
```

构建结果服务于模型层 activation 通信，不用于 CTDG feature fetch。

### CTDG SpatialRouteData

`SpatialRouteData` 是 CTDG 采样之后的 feature/memory fetch route。它不应作为主预构建 route，因为 CTDG 的 remote read set 取决于在线 sampler、负采样和 target set：

```text
send_index    torch.int64[S]     # 本 rank 本地可读 store row，按目标 part 排序
send_ptr      torch.int64[P + 1]
recv_ptr      torch.int64[P + 1]
recv_index    torch.int64[R]     # packed read/master DistIndex，按来源 part 排序
recv_node_ids optional torch.int64[R]
```

CTDG 可缓存 `SpatialRouteData`，但缓存是 runtime profile/cache，不是 prepare 必需 artifact：

```text
key = (placement_version, time_slice_id, root_hash/sample_hash)
value = SpatialRouteData
```

edge predict 在线负样本带来的 remote dst、node predict 动态 target set 带来的 remote context node 都进入本次采样输出的 `remote_read_index`，再即时构建或命中缓存得到 `SpatialRouteData`。

### MemoryRouteData

memory route 也提前构建，但需要拆成“最小必需字段”和“可选加速字段”，避免冗余。prepare 已知每个 time-slice 中各节点最后时间戳 message 位于哪个分区，以及该发送给哪些 owner/master。

最小必需字段：

```text
unique_index       torch.int64[D]      # packed master DistIndex
cand_pos           torch.int64[D, K]   # cand_pos[:,0] 是最后时间戳 message
send_ptr           torch.int64[P + 1]
recv_ptr           torch.int64[P + 1]
recv_index         torch.int64[R]      # packed master DistIndex
```

字段含义：

- `unique_index`：本 slice 需要读/写 memory 的唯一节点，已 packed 到 master row 和 master part。它替代 `unique_nodes + owner + local_id`。
- `cand_pos`：每个唯一节点对应的 event/message 位置。`cand_pos[:,0]` 是该 slice 内最大 timestamp 的 message，用于默认写回；`K>1` 只服务训练扰动、replay 或 debug。
- `send_ptr`：本 rank 按目标 master part 分组后，需要发送的更新切分。
- `recv_ptr`：本 rank 将从各 peer 接收的更新切分。
- `recv_index`：接收到的更新应写入哪个 master row。

可选字段：

```text
unique_nodes       optional int64[D]   # debug/兼容；热路径不需要
latest_msg_part    optional int16/int32[D] # 如果 unique_index.part 已可表达，则不存
replica_idx        optional torch.int64[H]
replica_send_ptr   optional torch.int64[P + 1]
replica_recv_ptr   optional torch.int64[P + 1]
```

冗余裁剪规则：

- 有 `unique_index` 就不在热路径使用 `unique_nodes/latest_msg_part`。
- 无热点副本时不保存 replica 三字段。
- `cand_pos` 默认 `K=1`；只有显式启用 perturb/replay 才扩展到 K>1。
- memory fetch route 和 writeback route 可共用 `unique_index/cand_pos`，但 send/recv ptr 方向不同，需要分开命名，避免误用。

### Route 构建约束

明确 prepare/runtime 分界：

prepare 阶段可以做：

- time-slice 内 `argsort/unique/bincount/scatter_add`。
- 构建 `chunk_load_by_slice`。
- 构建 DTDG `ChunkPropagationRoute` / `propagation_routes`。
- 构建 CTDG `memory_writeback_route` 所需的稳定 message-owner 关系；CTDG feature fetch 的 `SpatialRouteData` 不在 prepare 中全量预构建。
- 构建 packed `master_dist_index/read_dist_index` 和 CSR artifact。

runtime 热路径只允许做：

- DTDG 依据 `snapshot_id/layer_id` 取预构建 `ChunkPropagationRoute`。
- CTDG 根据 `CTDGSampleResult.remote_read_index` 即时构建或命中缓存 `SpatialRouteData`。
- 对在线负样本/动态 target/cache miss 生成本 batch fetch route。
- 以 rank/layer 为粒度的短循环。

runtime 禁止：

- event/edge 级 Python loop。
- 每 batch 全量 `unique/argsort`。
- 在 CTDG 中重建全 slice 图结构；只允许对采样后的 remote read set 建 fetch route。
- 反复构造 Python list 形式 route；除 `dist.all_to_all_single` API 需要 split sizes 时，其他均保持 tensor。

dynamic delta route 构建规则：

```text
input: packed_index[M] sorted or unsorted
part = dist_index_part(packed_index)
sort by part only if not already grouped
send_ptr = ind2ptr(part)
recv_ptr = all_to_all(send_counts)
recv_index = all_to_all(dist_index_loc/packed_index)
```

若 CTDG fetch route 的 M 长期过大，记录 profile，下一轮调整 time-slice、cache 策略或热点副本；不要把它误写成 DTDG `ChunkPropagationRoute`。

## 负样本采样

新增 `EdgePredictNegativeSampler`：

```yaml
sampler:
  neg_strategy: edge_predict_mixed
  neg_ratio: 1
  train_remote_dst_prob: 0.15
  test_policy: global_average
```

train：

```text
with probability 1 - train_remote_dst_prob:
    neg_dst ~ local_dst_pool[rank]
else:
    neg_dst ~ global_dst_pool
```

test/val：

- 使用全军平均的 `global_dst_pool`。
- 每个 rank 使用同一 base seed 和不同 batch offset，保证分布一致但样本不必完全相同。
- 若 neg dst 不可本地读，追加 dynamic delta fetch route。

可选输出 `neg_weight = p_test(neg_dst) / p_train(neg_dst)`，第一版默认不启用 loss reweight。

## 异步通信管理

`CommPipeline` 扩展为 handle 化接口：

```text
submit_fetch(plan, tensors)      -> CommHandle
submit_propagation(route, x)     -> RouteHandle
submit_state_sync(plan, state)   -> CommHandle
submit_grad_sync(bucket)         -> GradHandle
await(handle)                    -> result
drain(channel=None)
```

通道：

- `fetch`：feature/memory read。
- `propagation`：DTDG model layer autograd all-to-all。
- `state_sync`：memory/mailbox 写回 owner/master/replica。
- `grad_sync`：参数梯度同步。

使用三类 stream：

```text
compute_stream
fetch_stream
comm_stream
```

典型流水：

```text
prefetch batch[t+1] feature/memory
compute  batch[t]
submit   batch[t] state writeback + grad sync
await    batch[t+1] only when materialize/model needs remote data
```

CTDG 采样可在 native sampler/CPU worker 中并行，feature/memory all-to-all 在 GPU stream 上异步。DTDG route 参考 FlareDTDG：`send()` 启动 async all-to-all，`recv()` 在真正需要 remote tensor 时 wait，backward 反向 all-to-all。

DTDG 模型层嵌入方式：

```text
GraphConv/Attention layer:
  local h
  key = RouteSendFunction.apply(h, route)
  h_all = RouteRecvFunction.apply(key)  # wait point, returns local+remote
  compute message passing on h_all

backward:
  RouteRecvFunction.backward submits grad all-to-all
  RouteSendFunction.backward waits and scatters grad to local h
```

`RecvFunction` 必须嵌入模型层的 activation 获取位置，而不是放在 dataloader 或 train step 外层。这样 backward 才能和 PyTorch autograd 图一致，并复现 FlareDTDG 的通信/计算重叠。

## 异步梯度同步与权重备份

支持：

- `ddp_sync`：baseline。
- `ordered_async`：默认第一版，forward/prepare 可 overlap 上一个 all-reduce，但 optimizer step 按 version 顺序 apply。
- `async_stale`：后续扩展，允许 `staleness <= K`。

`WeightVersionManager`：

```text
active_version
backup_slots[K + 1]        # flat parameter buffer 或 state_dict shard
grad_bucket.version
optimizer_apply_version
```

第一版 ordered async：

1. step `v` forward 前保存 flat param buffer 到 slot `v % slots`。
2. backward hook 写入 `GradBucket(v)`。
3. bucket 满后 `all_reduce(async_op=True)`。
4. step `v+1` 的 sample/fetch/forward 可以和 `GradBucket(v)` overlap。
5. optimizer apply 必须等待所有 `< v` bucket 完成，按顺序 step。

memory/mailbox 不属于模型参数，走 `state_sync`。同节点多更新按最大 timestamp reduce；热点节点先写 master，再异步 fanout replica。

## Train Step

### CTDG Edge Predict

```text
for unit in loader.iter_train_units():
    view = unit.payload  # EventView with time_slice_id
    pos_edges = canonical events for this rank/slice
    neg_edges = neg_sampler.sample(split="train")

    sample_handle = event_engine.submit_sample(view, pos_edges, neg_edges)
    fetch_route   = build_or_get_spatial_route(sampled.remote_read_index)
    fetch_handle  = comm.submit_fetch(fetch_route)

    sampled = sample_handle.result()
    remote  = await fetch_handle
    batch   = materialize(sampled, remote, compact ids)

    out  = model(batch)
    loss = edge_predict_loss(out, batch)
    loss.backward()

    state = memory_updater(...)
    comm.submit_state_sync(prebuilt_memory_route, state)

    grad_sync.maybe_submit()
    grad_sync.maybe_apply_ready(optimizer)
```

### CTDG Node Predict

```text
for unit in loader.iter_train_units():
    view = unit.payload  # EventView with time_slice_id
    targets = target store for this rank/slice

    sample_handle = event_engine.submit_sample(view, root_nodes=targets)
    fetch_route   = build_or_get_spatial_route(sampled.remote_read_index)
    fetch_handle  = comm.submit_fetch(fetch_route)

    sampled = sample_handle.result()
    remote  = await fetch_handle
    batch   = materialize(sampled, remote, targets)

    out  = model(batch)
    loss = node_predict_loss(out, batch)
    loss.backward()

    state = memory_updater(...) if model has temporal memory else None
    comm.submit_state_sync(prebuilt_memory_route, state)

    grad_sync.maybe_submit()
    grad_sync.maybe_apply_ready(optimizer)
```

### DTDG Edge Predict

```text
for unit in loader.iter_train_units():
    block = unit.payload
    neg_edges = neg_sampler.sample(split="train")

    h = model.encode(block)  # route.async_forward inside layers
    out = edge_head(h, pos_edges, neg_edges, id_map=block.id_map)
    loss.backward()          # route backward returns remote grad to owner

    grad_sync.maybe_submit()
    grad_sync.maybe_apply_ready(optimizer)
```

### DTDG Node Predict

```text
for unit in loader.iter_train_units():
    block = unit.payload
    targets = block.target_nodes

    h = model.encode(block)  # route.async_forward inside layers
    out = node_head(h, targets, id_map=block.id_map)
    loss = node_predict_loss(out, block.labels)
    loss.backward()

    grad_sync.maybe_submit()
    grad_sync.maybe_apply_ready(optimizer)
```

## Eval/Test Step

test 不更新参数。edge predict 负采样使用全军平均 dst；node predict 不做负采样，只按 target split 评估：

```text
for unit in loader.iter_eval_units(split="test"):
    neg = neg_sampler.sample(split="test", policy="global_average")
    fetch remote neg dst if needed
    with torch.no_grad():
        out = model(batch)
        metrics.update(out)
```

CTDG memory policy：

```yaml
eval:
  memory_policy: temporal_replay  # temporal_replay | frozen
  negative_policy: global_average
```

`temporal_replay` 按时间推进 memory 但不反传、不更新参数；`frozen` 使用 checkpoint memory。

## 模型层目录

模型层单独管理，backend 不放模型实现：

```text
starry_unigraph/models/
  layers/
    route.py              # ChunkPropagationRoute adapter
    graph_conv.py          # route-aware GCN/SAGE/attention
    temporal_memory.py     # TGN/JODIE/DyRep/APAN memory updater
    recurrent.py           # GRU/LSTM/EvolveGCN/MPNN-LSTM cells
    scoring.py             # dot/bilinear/mlp edge scorers
  encoders/
    ctdg_tgn.py
    ctdg_jodie.py
    ctdg_dyrep.py
    ctdg_apan.py
    dtdg_gcn.py
    dtdg_tgcn.py
    dtdg_evolvegcn.py
    dtdg_mpnn_lstm.py
  heads/
    edge_predict.py
    node_classify.py
    node_regress.py
```

`starry_unigraph/models/task_head.py` 只保留兼容导入，真实实现迁移到 `models/heads/*`。

统一接口：

```python
class ChunkModel(nn.Module):
    def encode(self, batch_or_unit):
        ...

    def forward(self, batch_or_unit):
        emb = self.encode(batch_or_unit)
        return self.task_head(emb, batch_or_unit)
```

edge predict head 必须支持 compact id：

```python
class ChunkEdgePredictHead(nn.Module):
    def forward(self, embeddings, pos_src, pos_dst, neg_src, neg_dst, id_map=None):
        ...
```

`id_map` 是 global/packed id 到 batch-local row 的映射；不能默认用 global id 直接索引 embedding。

node predict head 同样必须支持 compact id：

```python
class ChunkNodePredictHead(nn.Module):
    def forward(self, embeddings, target_nodes, id_map=None):
        ...
```

node predict 包括 node classification 和 node regression。`target_nodes` 可以是 global id、packed DistIndex 或 batch-local row；进入 head 前统一通过 `id_map` 转成 compact row，避免全局 embedding 假设。

## Artifact Layout

```text
prepared_dir/
  meta.json
  placement.pth
  events/
    canonical_events.pth
    dst_pool_global.pth
    dst_pool_rank_{rank:03d}.pth
  targets/
    node_targets.pth
    target_pool_rank_{rank:03d}.pth
  chunks/
    chunk_assignment.pth
    chunk_load_by_slice.pth
    chunk_graph.pth
  indices/
    master_dist_index.pth
    read_dist_index_{rank:03d}.pth
    temporal_index_{rank:03d}.pth
    snapshot_index_{rank:03d}.pth
  partitions/
    manifest.json
    part_{rank:03d}.pth
  routes/
    mem_routes_{rank:03d}.pth
    propagation_routes_{rank:03d}.pth
  route_cache/                 # optional runtime cache, not required prepare output
    spatial_fetch_routes_{rank:03d}.pth
  cpu_layout_{rank:03d}.pth
```

manifest 必须记录：

```text
placement_version
partition_algo
chunks_per_partition
time_slice_policy
hot_policy
event_assignment_policy
target_assignment_policy
negative_sampling_policy
route_schema_version
num_parts
num_chunks
num_slices
```

## 与参考项目的映射

FlareDTDG：

- 采纳 autograd `RouteSendFunction/RouteRecvFunction`，并把 `RecvFunction` 嵌入 DTDG 模型层的 remote activation 获取点。
- 采纳 `AsyncModule.layerwise` 的协程调度，让多层/多 snapshot 的 route send 先发出，recv 在真实依赖点等待。
- 采纳 STGraphLoader/Dataloader 的高性能链路：一次性 pin/copy block、chunk order/chunk decay、按 snapshot/chunk 切片而不是 step 内重建图。
- route 返回需要支持 local+remote append，以满足 DTDG block 的 compact src row 索引。
- route 对象落到 `models/layers/route.py` 和 chunk runtime adapter，不依赖 Flare 包。

MemShare-public：

- 采纳 packed `DistIndex`、在线负采样、异步 feature/memory all-to-all、timestamp reduce、hot/shared node cache 思路。
- 采纳 dataloader pipeline 的阶段划分：submit sampling future -> async feature/memory fetch -> next iteration wait/patch。
- 通信逻辑下沉到 `CommPipeline`，但保留 MemShare 高性能链路的“一拍 ahead”队列和 pinned buffer 复用。

## 对齐 FlareDTDG/MemShare 的精度与效率风险

同一数据、模型、负采样和评估协议下，最终精度和训练效率需要追平 FlareDTDG 与 MemShare-public。主要风险与约束：

精度风险：

- train 使用 local-biased negative，test 使用 global-average negative，分布不一致可能影响 AUC/AP。需要记录 `neg_weight`，并提供开关做 importance reweight；默认实验必须同时报告 biased/unbiased 设置。
- CTDG memory 异步写回可能乱序。必须按 timestamp reduce，旧 ts 更新丢弃；eval 的 `temporal_replay/frozen` 语义必须和 MemShare 对齐。
- 热点副本读到 stale memory 会影响 temporal model。热点 replica 需要 version/ts，超过 staleness 阈值时强制读 master。
- DTDG route backward 若 scatter/index_add 语义和 Flare 不一致，会造成梯度偏差。需要单测对齐单机全图梯度。
- node predict target_owner 只决定 loss 归属，不能改变 label split 或 target set；否则评估不可比。

效率风险：

- runtime 若每 batch 做全量 `unique/argsort`，会输给 MemShare dataloader pipeline。必须把 full route 放 prepare，runtime 只处理 delta。
- `SpatialRouteData` 与 `ChunkPropagationRoute` 混用会导致 DTDG activation 通信跑到 dataloader，失去 autograd overlap。必须在模型层使用 `RecvFunction`。
- packed DistIndex 如果解包过于频繁，会抵消减少冗余的收益。热路径应用 fused op 或一次解包出 `part/loc`。
- memory route 字段过多会增加 H2D 和 cache 压力。默认只保存最小必需字段，debug 字段不进 GPU hot path。
- CTDG fetch route 长期过大说明采样 remote read set、time-slice 或热点副本策略不合理；不要用预构建 `SpatialRouteData` 掩盖问题。profile 中必须记录 remote/local read 比例和 fetch route 构建耗时。
- Python rank/layer 外循环可以接受，edge/event/node 级循环不可接受；否则无法追平参考项目。

验收基线：

- DTDG：同配置下 route forward/backward 时间、epoch time、显存峰值对齐 FlareDTDG；单机/多机 loss 曲线与全图 baseline 误差在可解释范围内。
- CTDG：同配置下 sampler+fetch pipeline throughput、memory sync 时间、AUC/AP 对齐 MemShare-public；负采样协议一致时指标不可显著下降。
- 统一 chunk：开启 packed DistIndex、DTDG `ChunkPropagationRoute` 预构建和 CTDG fetch route cache 后，runtime profile 中全量 `unique/argsort` 不应出现在每 step 主路径。

## 实现拆分

1. 固化 `time_ptr`：CTDG 接自适应切分，DTDG 接 snapshot/window。
2. 把 `chunk_to_nodes` list 改为 CSR artifact，并输出 `chunk_load_by_slice`。
3. 完善 `chunk_metis_balance`：先全图切 P*K chunk，再做负载向量均衡。
4. 引入 `master_dist_index/read_dist_index`，减少 route 与 batch 冗余字段。
5. 改 `SpatialRouteData/MemoryRouteData`：优先传 packed index；`SpatialRouteData` 由 CTDG sampled remote set 构建/缓存，`MemoryRouteData` 只保留稳定写回关系。
6. 实现 `EdgePredictNegativeSampler` 的 train local-biased 与 test global_average。
7. 预构建 DTDG `ChunkPropagationRoute`，并修正 append local+remote 与 backward gradient scatter。
8. 建 `models/layers|encoders|heads` 目录，迁移 task head 和 route-aware layer。
9. 改 `CommPipeline` 为 handle + stream + drain 模式。
10. 实现 `WeightVersionManager` + ordered async grad sync。

## 性能约束

- runtime 禁止 event/edge 级 Python loop。
- route merge、dedup、global/packed id 到 compact row 映射应做 fused op 或单次 tensor op。
- 每个 batch 避免重复 `torch.unique/argsort`；这些优先在 prepare 或 route cache 中完成。
- `TensorData`/CSR 一次 pack，runtime 只切片。
- 只允许在 rank 数、layer 数、time-slice 数这种小维度上 Python loop。
